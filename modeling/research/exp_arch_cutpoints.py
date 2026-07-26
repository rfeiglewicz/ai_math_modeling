#!/usr/bin/env python3
"""
Third-stage exploration: minimal-cost correctly-rounded BF16 exp(x), x <= 0.

Central observation
-------------------
The 128 numbers stored in the threshold ladder do NOT have to be the true
breakpoints  thr[k] = 1 - log2((127.5+k)/128).  They only have to SEPARATE
the 1967 f-values that can actually occur.  Between the largest f that must
map to a count >= k and the smallest f that must map to a count < k there is
an open interval, and any cut inside it is equally correct.

That freedom is worth a lot:
  * the cut can be snapped to a coarse grid  -> fewer bits per entry
    (and, because every cut is a multiple of 2^-B, f itself only needs
     B fractional bits: truncation cannot move f across a cut)
  * the cut can be slid to minimise the spread of "deviation from a linear
    ladder"                                   -> fewer bits per entry again

The script sweeps (P, B), builds the cuts by construction, and re-verifies
the complete integer datapath exhaustively.

Datapath being verified
-----------------------
    prod  = (128 + mant) * Lp                 Lp = round(log2(e) * 2^P)
    S     = P + 7 - e                         binary point of prod
    I     = prod >> S                         -> exponent  = -I-1
    f_i   = (prod mod 2^S) >> (S - B)         -> B-bit truncated fraction
    cnt   = #{ k in 1..128 : f_i <= cut[k] }
    result = pack(-I-1, cnt)                  cnt==128 -> (exp+1, mant 0)
"""

from decimal import Decimal, getcontext
from fractions import Fraction as Fr
import math

getcontext().prec = 90

LOG2E = Fr(1 / Decimal(2).ln())
BIAS, MANT_W = 127, 7
MAX_FINITE = 0x7F7F


# ---------------------------------------------------------------- bf16 utils
def bf16_value(code: int) -> Fr:
    if code < 128:
        return Fr(code) * Fr(2) ** -133
    return Fr(128 + (code & 0x7F), 128) * Fr(2) ** ((code >> MANT_W) - BIAS)


def bf16_round(v: Fr) -> int:
    if v <= 0:
        return 0
    lo, hi = 0, MAX_FINITE
    if v >= bf16_value(hi):
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bf16_value(mid) <= v:
            lo = mid
        else:
            hi = mid - 1
    dl, dh = v - bf16_value(lo), bf16_value(lo + 1) - v
    if dh < dl:
        return lo + 1
    if dh == dl:
        return lo + (lo & 1)
    return lo


def exp_exact(u: Fr) -> Fr:
    return Fr((-Decimal(u.numerator) / Decimal(u.denominator)).exp())


E_MIN, E_MAX = -9, 7
INPUTS = [(e, m) for e in range(E_MIN, E_MAX + 1) for m in range(128)]
val = lambda e, m: Fr(128 + m, 128) * Fr(2) ** e
REF = {k: bf16_round(exp_exact(val(*k))) for k in INPUTS}
NORMAL = [k for k in INPUTS if REF[k] >= 128]
TAIL = [k for k in INPUTS if 0 < REF[k] < 128]
ZERO = [k for k in INPUTS if REF[k] == 0]


def ref_count(key):
    """Required threshold count cnt for a normal-output input."""
    code = REF[key]
    e_out = (code >> MANT_W) - BIAS
    m_out = code & 0x7F
    # cnt == 128 was folded into (e_out+1, 0) by the packer
    return (m_out, e_out)


# ---------------------------------------------------------------------------
print("=" * 78)
print("A. Separability of the real f-values  (minimal grid B, per P)")
print("=" * 78)
print(f"{'P':>4} {'B':>4} {'sep?':>6} {'cutROM(b)':>10} {'devW':>5} "
      f"{'devROM(b)':>10}")


def datapath_f(P, B):
    """Return {key: (I, f_i)} for all normal inputs."""
    Lp = int(LOG2E * 2 ** P + Fr(1, 2))
    out = {}
    for e, m in NORMAL:
        prod = (128 + m) * Lp
        S = P + 7 - e
        I = prod >> S
        f_i = (prod & ((1 << S) - 1)) >> (S - B)
        out[(e, m)] = (I, f_i)
    return out


def required_cnt(P, B, fmap):
    """cnt that the ladder must produce for each input, from the reference."""
    need = {}
    ok = True
    for key in NORMAL:
        I, _ = fmap[key]
        m_out, e_out = ref_count(key)
        # exponent produced by the integer path
        if -I - 1 == e_out:
            cnt = m_out
        elif -I - 1 == e_out - 1:
            cnt = 128            # ladder must overflow the mantissa
        else:
            ok = False           # integer exponent is unrecoverable
            break
        need[key] = cnt
    return need if ok else None


def build_cuts(fmap, need):
    """
    cut[k] must satisfy  f_i <= cut[k]  <=>  need >= k.
    Returns (cuts, lo, hi) or None when not separable.
    lo[k] = max f_i with need >= k ; hi[k] = min f_i with need < k
    """
    NEG = -1
    INF = 1 << 62
    lo = [NEG] * 129
    hi = [INF] * 129
    for key, cnt in need.items():
        f_i = fmap[key][1]
        for k in range(1, 129):
            if cnt >= k:
                if f_i > lo[k]:
                    lo[k] = f_i
            else:
                if f_i < hi[k]:
                    hi[k] = f_i
    for k in range(1, 129):
        if lo[k] >= hi[k]:
            return None
    return lo, hi


best_cfg = None
results = []
for P in range(18, 33):
    for B in range(8, P - 6):
        fmap = datapath_f(P, B)
        need = required_cnt(P, B, fmap)
        if need is None:
            continue
        built = build_cuts(fmap, need)
        if built is None:
            continue
        lo, hi = built
        # --- raw cut ROM: cut[k] = lo[k] (any value in [lo, hi) works)
        cut_rom = 128 * B
        # --- deviation-from-linear encoding, cut slid to minimise the range
        dlo, dhi = [], []
        for k in range(1, 129):
            lin = ((128 - k) << B) // 128
            l = lo[k] if lo[k] >= 0 else 0
            h = hi[k] - 1 if hi[k] < (1 << 62) else (1 << B) - 1
            dlo.append(l - lin)
            dhi.append(h - lin)
        W = max(0, max(dlo) - min(dhi))
        devw = max(1, (W).bit_length())
        dev_rom = 128 * devw
        results.append((P, B, cut_rom, devw, dev_rom))
        print(f"{P:>4} {B:>4} {'yes':>6} {cut_rom:>10} {devw:>5} {dev_rom:>10}")
        if best_cfg is None or dev_rom < best_cfg[4]:
            best_cfg = (P, B, cut_rom, devw, dev_rom)
        break                     # smallest B for this P

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("B. Exhaustive re-verification of the winning configuration")
print("=" * 78)

P, B, cut_rom, devw, dev_rom = best_cfg
print(f"chosen: P={P}, B={B}, cut ROM 128x{B}={cut_rom} b, "
      f"deviation ROM 128x{devw}={dev_rom} b")

fmap = datapath_f(P, B)
need = required_cnt(P, B, fmap)
lo, hi = build_cuts(fmap, need)
CUT = [0] * 129
for k in range(1, 129):
    CUT[k] = lo[k] if lo[k] >= 0 else -1

Lp = int(LOG2E * 2 ** P + Fr(1, 2))
errors = 0
for e, m in NORMAL:
    prod = (128 + m) * Lp
    S = P + 7 - e
    I = prod >> S
    f_i = (prod & ((1 << S) - 1)) >> (S - B)
    cnt = 0
    for k in range(1, 129):
        if f_i <= CUT[k]:
            cnt += 1
    e_out, m_out = -I - 1, cnt
    if m_out == 128:
        m_out, e_out = 0, e_out + 1
    if (BIAS + e_out) * 128 + m_out != REF[(e, m)]:
        errors += 1
print(f"normal-output inputs verified : {len(NORMAL)}, mismatches = {errors}")

print(f"log2(e) constant  Lp = 0x{Lp:X}  ({P} frac bits, "
      f"{bin(Lp).count('1')} ones)")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("C. Killing the 128-wide comparator array")
print("=" * 78)
print("Use the top L bits of f to look up a candidate count, then correct it")
print("with a handful of comparisons against neighbouring cuts.")
print(f"{'L':>3} {'candROM(b)':>11} {'max corr':>9} {'cmp needed':>11}")

CAND = {}
for L in range(6, 13):
    worst = 0
    cand = {}
    for key in NORMAL:
        f_i = fmap[key][1]
        idx = f_i >> (B - L)
        c = need[key]
        if idx in cand:
            cand[idx] = (min(cand[idx][0], c), max(cand[idx][1], c))
        else:
            cand[idx] = (c, c)
    worst = max(h - l for l, h in cand.values())
    print(f"{L:>3} {(1 << L) * 8:>11} {worst:>9} {worst:>11}")
    if worst <= 1:
        CAND = (L, cand)
        print(f"   -> L={L}: ROM 2^{L} x 8 bits gives the count up to +/-{worst};")
        print(f"      a single comparison against one cut finishes the job")
        break

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("C2. Shrinking the candidate ROM (deviation from a linear ladder)")
print("=" * 78)
L, cand = CAND
dlo, dhi = [], []
for idx in range(1 << L):
    lin = 128 - ((idx * 128) >> L)
    if idx in cand:
        l, h = cand[idx]
    else:                       # unused bucket: any value is fine
        continue
    dlo.append(l - lin)
    dhi.append(h - lin)
print(f"L = {L}, buckets used = {len(cand)} / {1 << L}")
print(f"base-count deviation from linear : {min(dlo)} .. {max(dhi)}")
span = max(dhi) - min(dlo)
cw = max(1, span.bit_length())
print(f"  -> {cw} bits/entry  =>  candidate ROM {(1 << L) * cw} bits "
      f"(instead of {(1 << L) * 8})")
print(f"  correction term is 0/1, resolved by ONE compare against cut[base+1]")
print(f"  the cut ROM is read once (1 port), not 128 times")

CAND_ROM = (1 << L) * cw
print()
print(f"logic estimate (Xilinx-class LUT6):")
print(f"  8x{P} constant multiplier (CSD)   ~ {bin(Lp).count('1') - 1} adders")
print(f"  barrel shift by (P+7-e), 17 cases ~ small mux")
print(f"  candidate ROM 2^{L} x {cw}          ~ {(1 << L) * cw // 64} LUT6")
print(f"  cut ROM 128 x {devw} (1 read port)   ~ {128 * devw // 64} LUT6")
print(f"  one {B}-bit comparator             ~ {B // 2} LUT6")
print(f"  (previous variant needed 128 x {B}-bit comparators "
      f"~ {128 * B // 2} LUT6)")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("D. Tail / zero handling")
print("=" * 78)
print(f"non-zero subnormal outputs : {len(TAIL)}  -> patch ROM {len(TAIL) * 7} b")
print(f"flush-to-zero inputs       : {len(ZERO)}  -> single compare on (e,mant)")
zmin = min(val(*k) for k in ZERO)
print(f"zero threshold             : x <= -{float(zmin):.6f}")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("E. Stored-bit ranking")
print("=" * 78)
tail_bits = len(TAIL) * 7
rows = [
    ("full LUT (2176 x 16)", 2176 * 16),
    ("current hybrid", 17198),
    ("true thresholds, P=26/F=18", 2304 + tail_bits),
    (f"optimal cuts, P={P}/B={B}", cut_rom + tail_bits),
    (f"optimal cuts + linear deviation ({devw} b/entry)", dev_rom + tail_bits),
    (f"  + candidate ROM 2^{L} (kills 128 comparators)",
     dev_rom + tail_bits + CAND_ROM),
]
for name, bits in rows:
    print(f"  {name:<48} {bits:>7} b   ({bits / 8:.0f} B)")
print()
print(f"  best vs hybrid : {100 * (1 - (dev_rom + tail_bits) / 17198):.1f} % fewer stored bits")
print(f"  best vs LUT    : {100 * (1 - (dev_rom + tail_bits) / (2176 * 16)):.1f} % fewer stored bits")
