#!/usr/bin/env python3
"""
Second-stage exploration: how cheap can the datapath get if the few
table-maker's-dilemma cases are handled by an exception patch instead of
by widening the whole datapath?

Architecture under test
-----------------------
    t     = (128+mant) * Lp        (constant multiplier, Lp = round(log2e * 2^P))
    I     = t >> (P+7-e)           (output exponent = -I-1)
    f     = low bits of t, kept to F fractional bits
    m_out = #{k in 1..128 : f <= thr[k]}     (monotone threshold ladder, 128 entries)
    + optional exception patch for inputs where the truncation flips one code
"""

from decimal import Decimal, getcontext
from fractions import Fraction as Fr
import math
from collections import Counter

getcontext().prec = 90
LOG2E = Fr(1 / Decimal(2).ln())
BIAS, MANT_W = 127, 7


def bf16_value(code):
    if code < 128:
        return Fr(code) * Fr(2) ** -133
    return Fr(128 + (code & 0x7F), 128) * Fr(2) ** ((code >> 7) - BIAS)


def bf16_round(v):
    if v <= 0:
        return 0
    lo, hi = 0, 0x7F7F
    if v >= bf16_value(hi):
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bf16_value(mid) <= v:
            lo = mid
        else:
            hi = mid - 1
    dl, dh = v - bf16_value(lo), bf16_value(lo + 1) - v
    return lo + 1 if dh < dl else (lo + (lo & 1) if dh == dl else lo)


def exp_exact(u):
    return Fr((-Decimal(u.numerator) / Decimal(u.denominator)).exp())


def log2_fr(q):
    return Fr((Decimal(q.numerator) / Decimal(q.denominator)).ln() / Decimal(2).ln())


INPUTS = [(e, m) for e in range(-9, 8) for m in range(128)]
val = lambda e, m: Fr(128 + m, 128) * Fr(2) ** e
REF = {k: bf16_round(exp_exact(val(*k))) for k in INPUTS}
NORMAL = [k for k in INPUTS if REF[k] >= 128]
THR = [1 - log2_fr(Fr(255 + 2 * k, 256)) for k in range(129)]

# ---------------------------------------------------------------- margins
print("=" * 78)
print("A. Distribution of the distance |f - nearest breakpoint|")
print("=" * 78)
marg = {}
for k in NORMAL:
    t = val(*k) * LOG2E
    f = t - int(math.floor(t))
    marg[k] = min(abs(f - THR[j]) for j in range(1, 129))
hist = Counter(max(0, math.ceil(-math.log2(float(d)))) for d in marg.values())
print(f"{'margin < 2^-b':>14} {'#inputs':>9}")
cum = 0
for b in range(4, 24):
    cum = sum(v for kk, v in hist.items() if kk >= b)
    if cum:
        print(f"{'2^-'+str(b):>14} {cum:>9}")
print(f"total normal-output inputs: {len(NORMAL)}")

# ------------------------------------------------- datapath + exceptions
print()
print("=" * 78)
print("B. Errors vs datapath precision, and cost of patching them")
print("=" * 78)


def run(P, F, mode="trunc"):
    Lp = int(LOG2E * 2 ** P + Fr(1, 2))
    if mode == "round":
        thr_i = [int(THR[j] * 2 ** F + Fr(1, 2)) for j in range(129)]
    else:
        thr_i = [math.floor(THR[j] * 2 ** F) for j in range(129)]
    bad = []
    for e, m in NORMAL:
        prod = (128 + m) * Lp
        S = P + 7 - e
        I = prod >> S
        f_i = (prod & ((1 << S) - 1)) >> (S - F)
        if mode == "round":
            f_i = ((prod & ((1 << S) - 1)) + (1 << (S - F - 1))) >> (S - F)
        cnt = sum(1 for j in range(1, 129) if f_i <= thr_i[j])
        e_out, m_out = -I - 1, cnt
        if m_out == 128:
            m_out, e_out = 0, e_out + 1
        code = (BIAS + e_out) * 128 + m_out
        if code != REF[(e, m)]:
            bad.append((e, m, code - REF[(e, m)]))
    return bad


print(f"{'P':>3} {'F':>3} {'mode':>6} {'#wrong':>7} {'deltas':>16} "
      f"{'thrROM':>7} {'patch bits':>10} {'total':>7}")
for F in (10, 11, 12, 13, 14, 15, 16, 18, 20, 24):
    for mode in ("trunc", "round"):
        P = F + 8
        bad = run(P, F, mode)
        d = sorted(set(x[2] for x in bad))
        rom = 128 * F
        patch = len(bad) * (11 + 1)          # 11-bit input code + direction bit
        print(f"{P:>3} {F:>3} {mode:>6} {len(bad):>7} {str(d):>16} "
              f"{rom:>7} {patch:>10} {rom+patch:>7}")

# --------------------------------------------------- exception locations
print()
print("=" * 78)
print("C. Where the exceptions live (P=F+8, F=14, round)")
print("=" * 78)
bad = run(22, 14, "round")
print(f"count = {len(bad)}")
for e, m, d in bad:
    print(f"  exp={e:>3} mant={m:>4}  ref=0x{REF[(e,m)]:04X}  delta={d:+d}")

# --------------------------------------------- threshold ROM compression
print()
print("=" * 78)
print("D. Compressing the 128 thresholds themselves")
print("=" * 78)
for F in (14, 16, 18, 24):
    thr_i = [math.floor(THR[j] * 2 ** F) for j in range(1, 129)]
    lin = [((128 - j) << F) // 128 for j in range(1, 129)]
    dev = [thr_i[i] - lin[i] for i in range(128)]
    w = max(dev).bit_length() + 1 if max(dev) > 0 else 1
    rng = max(dev) - min(dev)
    print(f"  F={F:>2}: raw 128x{F}={128*F:>5} b | "
          f"deviation range {min(dev)}..{max(dev)} -> {rng.bit_length()} b/entry "
          f"= {128*rng.bit_length():>5} b (+ subtractor)")
