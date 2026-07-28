#!/usr/bin/env python3
"""Minimax polynomial study for the BF16 exp(x) back end.

Front end is identical to the cut-point ladder:
    t = -x * log2(e),  I = floor(t),  f = frac(t)
    exp(x) = 2^(-I-1) * 2^(1-f)

Only the back end differs: instead of classifying f against cut points, a
minimax polynomial P(f) ~= 2^(1-f) is evaluated with Horner and the result is
rounded to the 7-bit BF16 mantissa.

This script answers: what polynomial degree is required for bit-exact
(<= 0.5 ULP) results, and how many inputs a degree-3 polynomial gets wrong.
"""

import math

import numpy as np

LOG2E = math.log2(math.e)

MIN_EXP, MAX_EXP = -9, 7          # unbiased BF16 exponent bins that matter
ZERO_EXP, ZERO_MANT_LO = 6, 58    # everything past this flushes to zero
TAIL_EXP, TAIL_MANT_LO = 6, 47    # subnormal tail, excluded from the ladder


def target(f):
    """2^(1-f), the normalized mantissa value, on f in [0, 1)."""
    return np.power(2.0, 1.0 - f)


def remez(func, deg, a=0.0, b=1.0, iters=80, grid=400_001):
    """Remez exchange -> minimax polynomial coefficients (ascending order)."""
    n = deg + 2
    k = np.arange(n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * (n - 1 - k) / (n - 1))
    xx = np.linspace(a, b, grid)
    fxx = func(xx)
    coeffs = None

    for _ in range(iters):
        vander = np.vander(x, deg + 1, increasing=True)
        alt = ((-1.0) ** np.arange(n)).reshape(-1, 1)
        sol = np.linalg.solve(np.hstack([vander, alt]), func(x))
        coeffs = sol[: deg + 1]

        err = np.polyval(coeffs[::-1], xx) - fxx
        slope = np.sign(np.diff(err))
        turning = np.where(np.diff(slope) != 0)[0] + 1
        cand = np.concatenate(([0], turning, [grid - 1]))

        groups, cur = [], [cand[0]]
        for i in cand[1:]:
            if np.sign(err[i]) == np.sign(err[cur[-1]]):
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        peaks = np.array([max(g, key=lambda i: abs(err[i])) for g in groups])
        if len(peaks) < n:
            break

        vals = np.abs(err[peaks])
        start = max(range(len(peaks) - n + 1), key=lambda s: vals[s : s + n].min())
        new_x = xx[peaks[start : start + n]]
        if np.allclose(new_x, x, atol=1e-13):
            x = new_x
            break
        x = new_x

    err = np.polyval(coeffs[::-1], xx) - func(xx)
    return coeffs, float(np.max(np.abs(err)))


def ladder_inputs():
    """The (x, f) pairs the ladder back end actually has to handle."""
    out = []
    for exp in range(MIN_EXP, MAX_EXP + 1):
        for mant in range(128):
            if exp > ZERO_EXP or (exp == ZERO_EXP and mant >= ZERO_MANT_LO):
                continue
            if exp == TAIL_EXP and mant >= TAIL_MANT_LO:
                continue  # handled by the subnormal tail ROM in every variant
            x = -(1.0 + mant / 128.0) * (2.0 ** exp)
            t = -x * LOG2E
            out.append((x, t - math.floor(t)))
    return out


def reference_code(f):
    """Correctly rounded mantissa code: how many RNE midpoints lie above f."""
    lo, hi = 0, 129
    while lo < hi:                      # thr[k] is strictly decreasing in k
        mid = (lo + hi) // 2
        if mid == 0 or f <= 1.0 - math.log2((127.5 + mid) / 128.0):
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def main():
    pts = ladder_inputs()
    fs = np.array([f for _, f in pts])
    refs = np.array([reference_code(f) for f in fs])

    gaps = []
    for f in fs:
        k = reference_code(f)
        for kk in (k, k + 1):
            if 1 <= kk <= 128:
                gaps.append(abs(f - (1.0 - math.log2((127.5 + kk) / 128.0))))
    min_gap = min(gaps)

    print(f"inputs routed through the ladder : {len(pts)}")
    print(f"min |f - thr[k]|                 : {min_gap:.4e}  (~2^{math.log2(min_gap):.1f})")
    need = min_gap * 128.0 * math.log(2.0)
    print(f"=> required |P(f) - 2^(1-f)|     : < {need:.4e}  (~2^{math.log2(need):.1f})")
    print()

    print(" deg | minimax err | margin vs need | mult | wrong | max ULP")
    print("-----+-------------+----------------+------+-------+--------")
    for deg in range(1, 8):
        coeffs, err = remez(target, deg)
        vals = np.polyval(coeffs[::-1], fs)
        codes = np.floor(128.0 * (vals - 1.0) + 0.5).astype(int)
        wrong = int(np.count_nonzero(codes != refs))
        exact = 128.0 * (target(fs) - 1.0)
        max_ulp = float(np.max(np.abs(codes - exact)))
        verdict = "OK" if err < need else f"{err / need:8.1f}x too big"
        print(
            f" {deg:3d} | {err:11.3e} | {verdict:>14} | {deg:4d} | "
            f"{wrong:5d} | {max_ulp:6.3f}"
        )

    print()
    coeffs, err = remez(target, 3)
    print("degree 3 Horner coefficients for 2^(1-f):")
    print("    P(f) = ((c3*f + c2)*f + c1)*f + c0")
    for i, c in enumerate(coeffs):
        print(f"    c{i} = {c: .12f}")
    vals = np.polyval(coeffs[::-1], fs)
    codes = np.floor(128.0 * (vals - 1.0) + 0.5).astype(int)
    bad = np.nonzero(codes != refs)[0]
    print(f"\n    mismatching inputs: {len(bad)} of {len(pts)}")
    for i in bad[:12]:
        x = pts[i][0]
        print(
            f"      x = {x:14.7f}   f = {fs[i]:.9f}   "
            f"poly {codes[i]:3d} vs ref {refs[i]:3d}"
        )


if __name__ == "__main__":
    main()
