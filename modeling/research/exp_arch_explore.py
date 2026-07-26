#!/usr/bin/env python3
"""
Architecture exploration for correctly-rounded BF16 exp(x), x <= 0.

Goal: find the cheapest hardware structure that is bit-exact with RNE
(i.e. <= 0.5 ULP) over the whole BF16 input space.

Key idea under test:
    exp(x) = 2^(x*log2e) = 2^(-I) * 2^(-f),  I = floor(t), f = frac(t), t = -x*log2e
    -> output exponent comes from I  (pure integer arithmetic)
    -> output mantissa depends ONLY on f, shared across all input exponent bins
    -> the f -> mantissa map is a monotone step function with 128 breakpoints,
       so it can be evaluated by threshold comparison instead of a 2^F entry ROM.

Everything is checked exhaustively against a high precision reference.
"""

from decimal import Decimal, getcontext
from fractions import Fraction as Fr
import math

getcontext().prec = 90

LOG2E = Fr(1 / Decimal(2).ln())          # ~90 significant digits
BIAS = 127
MANT_W = 7
MIN_SUBNORMAL_EXP = -126 - MANT_W        # 2^-133

MAX_FINITE_CODE = 0x7F7F                 # largest finite positive bf16 payload


# --------------------------------------------------------------------------
# BF16 helpers (exact rational arithmetic)
# --------------------------------------------------------------------------
def bf16_value(code: int) -> Fr:
    """Value of a non-negative finite BF16 payload."""
    if code < 128:                                   # zero / subnormal
        return Fr(code) * Fr(2) ** MIN_SUBNORMAL_EXP
    e = (code >> MANT_W) - BIAS
    m = code & 0x7F
    return Fr(128 + m, 128) * Fr(2) ** e


def bf16_round(v: Fr) -> int:
    """Round a non-negative exact value to a BF16 payload, RNE."""
    if v <= 0:
        return 0
    lo, hi = 0, MAX_FINITE_CODE
    if v >= bf16_value(hi):
        return hi
    while lo < hi:                                   # largest code with value <= v
        mid = (lo + hi + 1) // 2
        if bf16_value(mid) <= v:
            lo = mid
        else:
            hi = mid - 1
    c = lo
    dl = v - bf16_value(c)
    dh = bf16_value(c + 1) - v
    if dh < dl:
        return c + 1
    if dh == dl:
        return c + (c & 1)                           # ties to even
    return c


def exp_exact(u: Fr) -> Fr:
    """exp(-u) as a very accurate rational (90 significant digits)."""
    d = (-Decimal(u.numerator) / Decimal(u.denominator)).exp()
    return Fr(d)


# --------------------------------------------------------------------------
# Input space:  x < 0, unbiased exponent in [-9, 7]
# --------------------------------------------------------------------------
E_MIN, E_MAX = -9, 7
INPUTS = [(e, m) for e in range(E_MIN, E_MAX + 1) for m in range(128)]


def input_value(e: int, m: int) -> Fr:
    return Fr(128 + m, 128) * Fr(2) ** e


print("=" * 78)
print("0. Reference table")
print("=" * 78)
REF = {}
for e, m in INPUTS:
    u = input_value(e, m)
    REF[(e, m)] = bf16_round(exp_exact(u))

n_zero = sum(1 for k, v in REF.items() if v == 0)
n_sub = sum(1 for k, v in REF.items() if 0 < v < 128)
n_norm = sum(1 for k, v in REF.items() if v >= 128)
print(f"inputs                     : {len(INPUTS)}")
print(f"  -> normal   result       : {n_norm}")
print(f"  -> subnormal result      : {n_sub}")
print(f"  -> zero     result       : {n_zero}")
sub_or_zero = sorted(k for k, v in REF.items() if v < 128)
if sub_or_zero:
    print(f"  subnormal/zero occur for exponent bins: "
          f"{sorted(set(e for e, _ in sub_or_zero))}")
    e6 = [m for e, m in sub_or_zero if e == 6]
    if e6:
        print(f"  in bin e=6 for mantissa codes {min(e6)}..{max(e6)} ({len(e6)} inputs)")


# --------------------------------------------------------------------------
# 1. Mantissa breakpoints in the f domain
# --------------------------------------------------------------------------
# mant = 2^(1-f) in [1,2];  m = RNE(128*(mant-1))
# m is the number of thresholds thr[1..128] that are >= f, where
#     thr[k] = 1 - log2((127.5+k)/128)
def log2_fr(q: Fr) -> Fr:
    d = (Decimal(q.numerator) / Decimal(q.denominator)).ln() / Decimal(2).ln()
    return Fr(d)


THR = [1 - log2_fr(Fr(255 + 2 * k, 256)) for k in range(129)]   # THR[0] > 1 unused
assert all(THR[k] > THR[k + 1] for k in range(128))

print()
print("=" * 78)
print("1. Breakpoint structure of the shared 2^-f table")
print("=" * 78)
print(f"thresholds                 : 128 (thr[1..128]), monotone decreasing")
print(f"thr[1]  = {float(THR[1]):.9f}   thr[128] = {float(THR[128]):.9f}")
# deviation from a purely linear (Schraudolph) threshold ladder
dev = [float(THR[k] - Fr(128 - k, 128)) for k in range(1, 129)]
print(f"deviation from linear ladder: [{min(dev):+.6f}, {max(dev):+.6f}]  "
       f"-> {max(abs(min(dev)), abs(max(dev)))*128:.2f} codes")


# --------------------------------------------------------------------------
# 2. Hardness: how close does f get to a breakpoint?
# --------------------------------------------------------------------------
print()
print("=" * 78)
print("2. Table-maker's-dilemma margin over the real input set")
print("=" * 78)

worst = None
for e, m in INPUTS:
    if REF[(e, m)] < 128:          # subnormal/zero handled separately
        continue
    t = input_value(e, m) * LOG2E
    f = t - int(math.floor(t))
    d = min(abs(f - THR[k]) for k in range(1, 129))
    if worst is None or d < worst[0]:
        worst = (d, e, m)

margin, we, wm = worst
bits_needed = -math.log2(float(margin))
print(f"min |f - thr| = {float(margin):.3e}  at (e={we}, mant={wm})")
print(f"  -> ~{bits_needed:.2f} fractional bits of f are strictly necessary")
print(f"  -> practical choice F = {math.ceil(bits_needed) + 2} bits")


# --------------------------------------------------------------------------
# 3. Exhaustive check of the proposed datapath
#    prod = (128+m) * Lp   (constant multiplier)
#    t    = prod >> (P+7-e)
#    m_out = popcount(f_trunc <= thr_int[k], k=1..128)
# --------------------------------------------------------------------------
print()
print("=" * 78)
print("3. Exhaustive bit-exactness sweep of the threshold datapath")
print("=" * 78)
print(" P = frac bits of the log2(e) constant, F = frac bits of f kept")
print(f"{'P':>4} {'F':>4} {'prod_w':>7} {'thr_ROM(b)':>11} {'errors':>7}")


def run_datapath(P: int, F: int, report_fail=False):
    Lp = round(float(LOG2E) * 2 ** P) if P < 50 else None
    Lp = (LOG2E * 2 ** P)
    Lp = int(Lp + Fr(1, 2))                       # round to nearest integer
    thr_int = [int(THR[k] * 2 ** F) if THR[k] > 0 else 0 for k in range(129)]
    thr_int = [int((THR[k] * 2 ** F).__floor__()) for k in range(129)]
    errors = []
    for e, m in INPUTS:
        ref = REF[(e, m)]
        if ref < 128:
            continue                              # subnormal path, see section 5
        prod = (128 + m) * Lp                     # integer
        S = P + 7 - e                             # binary point position in prod
        I = prod >> S
        frac_bits = prod & ((1 << S) - 1)
        f_int = frac_bits >> (S - F)              # keep F bits, truncate
        cnt = 0
        for k in range(1, 129):
            if f_int <= thr_int[k]:
                cnt += 1
        e_out = -I - 1
        m_out = cnt
        if m_out == 128:                          # mantissa rounded up to 2.0
            m_out = 0
            e_out += 1
        code = (BIAS + e_out) * 128 + m_out
        if code != ref:
            errors.append((e, m, code, ref))
    if report_fail and errors:
        for x in errors[:8]:
            print("   FAIL", x)
    return errors


best = None
for P in range(16, 34, 2):
    for F in range(10, 24, 2):
        if P < F + 8:
            continue
        errs = run_datapath(P, F)
        rom = 128 * F
        print(f"{P:>4} {F:>4} {8+P:>7} {rom:>11} {len(errs):>7}")
        if not errs and best is None:
            best = (P, F, rom)
    if best:
        break

if best:
    P, F, rom = best
    print()
    print(f"SMALLEST EXACT CONFIG: P={P} (log2e constant bits), F={F} (frac bits of f)")
    print(f"  threshold ROM        : 128 x {F} = {rom} bits")
    print(f"  constant multiplier  : 8 x {P} bits (multiply by a CONSTANT)")


# --------------------------------------------------------------------------
# 4. Alternative: direct 2^F-entry mantissa ROM (no comparators)
#    How many index bits until every bucket is unambiguous?
# --------------------------------------------------------------------------
print()
print("=" * 78)
print("4. Direct-indexed mantissa ROM instead of thresholds")
print("=" * 78)
fvals = {}
for e, m in INPUTS:
    if REF[(e, m)] < 128:
        continue
    t = input_value(e, m) * LOG2E
    f = t - int(math.floor(t))
    fvals[(e, m)] = (f, REF[(e, m)] & 0x7F)

for Fi in range(6, 20):
    buckets = {}
    ok = True
    for key, (f, mm) in fvals.items():
        b = int(f * 2 ** Fi)
        if buckets.setdefault(b, mm) != mm:
            ok = False
            break
    if ok:
        print(f"min index bits          : {Fi}  ->  ROM {2**Fi} x 7 = {2**Fi*7} bits")
        print(f"  (as deviation-from-linear: {2**Fi} x 4 = {2**Fi*4} bits + adder)")
        break


# --------------------------------------------------------------------------
# 5. Cost of the subnormal / zero tail
# --------------------------------------------------------------------------
print()
print("=" * 78)
print("5. Subnormal tail")
print("=" * 78)
tail = [(e, m) for (e, m) in INPUTS if REF[(e, m)] < 128 and REF[(e, m)] != 0]
if tail:
    es = sorted(set(e for e, _ in tail))
    print(f"non-zero subnormal outputs : {len(tail)} inputs, exponent bins {es}")
    ms = sorted(m for e, m in tail)
    print(f"mantissa codes             : {min(ms)}..{max(ms)}")
    seq = [REF[k] for k in sorted(tail)]
    d = [seq[i] - seq[i + 1] for i in range(len(seq) - 1)]
    print(f"output payloads            : {seq[0]} down to {seq[-1]}")
    print(f"successive deltas          : min {min(d)} max {max(d)} "
          f"-> threshold/RLE encodable: {set(d) <= {0,1}}")
    print(f"plain patch ROM            : {len(tail)} x 7 = {len(tail)*7} bits")


# --------------------------------------------------------------------------
# 6. CSD cost of the log2(e) constant multiplier
# --------------------------------------------------------------------------
print()
print("=" * 78)
print("6. Constant multiplier cost (log2 e)")
print("=" * 78)


def csd_terms(val: int) -> int:
    """Number of non-zero digits in the canonical signed digit recoding."""
    n, terms = val, 0
    while n:
        if n & 1:
            z = 2 - (n & 3)          # +1 or -1
            n -= z
            terms += 1
        n >>= 1
    return terms


for P in (16, 18, 20, 22, 24, 26):
    v = int(LOG2E * 2 ** P + Fr(1, 2))
    print(f"  P={P:>3}  const=0x{v:X}  binary ones={bin(v).count('1'):>3}  "
          f"CSD terms={csd_terms(v):>3}  -> ~{csd_terms(v)-1} adders")


# --------------------------------------------------------------------------
# 7. Payload comparison
# --------------------------------------------------------------------------
print()
print("=" * 78)
print("7. Stored-bit comparison")
print("=" * 78)
print(f"  full LUT                 : 2176 x 16 = {2176*16} bits")
print(f"  current hybrid           :            17198 bits")
if best:
    P, F, rom = best
    tail_bits = len(tail) * 7 if tail else 0
    print(f"  threshold architecture   : {rom} (thresholds) + {tail_bits} (subnormal tail)"
          f" = {rom + tail_bits} bits")
    print(f"  reduction vs hybrid      : {100*(1-(rom+tail_bits)/17198):.1f}%")
