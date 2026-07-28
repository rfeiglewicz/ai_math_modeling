#!/usr/bin/env python3
"""Can a degree-3 polynomial be RETUNED to be bit-exact for BF16 exp(x)?

poly_minimax_study.py showed the degree-3 minimax polynomial misses 24 of the
1967 ladder inputs.  But minimax optimizes the WRONG objective: we do not need
|P(f) - 2^(1-f)| to be small, we only need

    round(128 * (P(f) - 1))  ==  correct mantissa code

for the 1967 f values that actually occur.  That is exactly the same "slack"
argument that lets the cut-point ladder store approximate cut points.

Written as constraints, for every input i:

    1 + (M_i - 0.5)/128  <=  P(f_i)  <=  1 + (M_i + 0.5)/128

which is LINEAR in the coefficients.  Maximizing the slack t in

    lo_i + t  <=  P(f_i)  <=  hi_i - t

is a linear program.  If the optimum has t > 0 the degree is feasible and the
24 minimax failures need no fixup hardware at all.

The LP is run on the TRUNCATED f values the hardware actually produces, so
truncation error is absorbed into the solution rather than added on top.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")
from scipy.optimize import linprog  # noqa: E402

from poly_minimax_study import LOG2E, ladder_inputs, reference_code, remez, target  # noqa: E402

FRAC_BITS = 16          # f width in the cut-point ladder datapath
CODE = 1.0 / 128.0      # one mantissa code in value units


def build_points(frac_bits):
    """Distinct (f_truncated, reference_code) pairs seen by the back end."""
    seen = {}
    for _, f in ladder_inputs():
        fq = int(f * (1 << frac_bits))          # hardware truncates
        ref = reference_code(f)                 # reference uses the EXACT f
        seen.setdefault(fq, ref)
    fq = np.array(sorted(seen), dtype=np.float64)
    refs = np.array([seen[int(v)] for v in fq], dtype=np.float64)
    return fq / (1 << frac_bits), refs


def max_margin(fs, refs, deg):
    """LP: maximize the classification margin for a polynomial of `deg`.

    Returns (coefficients ascending, margin in mantissa-code units).
    """
    n = len(fs)
    vander = np.vander(fs, deg + 1, increasing=True)     # rows: [1, f, f^2, ...]
    lo = 1.0 + (refs - 0.5) * CODE
    hi = 1.0 + (refs + 0.5) * CODE

    # variables: [c_0 .. c_deg, t]     objective: maximize t
    a_ub = np.zeros((2 * n, deg + 2))
    b_ub = np.zeros(2 * n)
    a_ub[:n, : deg + 1] = -vander                 # -P(f) + t <= -lo
    a_ub[:n, -1] = 1.0
    b_ub[:n] = -lo
    a_ub[n:, : deg + 1] = vander                  #  P(f) + t <=  hi
    a_ub[n:, -1] = 1.0
    b_ub[n:] = hi

    obj = np.zeros(deg + 2)
    obj[-1] = -1.0
    bounds = [(-8.0, 8.0)] * (deg + 1) + [(None, None)]

    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None, None
    return res.x[: deg + 1], res.x[-1] / CODE     # margin in code units


def count_wrong(coeffs, fs, refs):
    vals = np.polyval(coeffs[::-1], fs)
    codes = np.floor(128.0 * (vals - 1.0) + 0.5)
    return int(np.count_nonzero(codes != refs))


def main():
    fs, refs = build_points(FRAC_BITS)
    print(f"distinct truncated f values : {len(fs)}  (f width {FRAC_BITS} bits)")
    print()

    print("How much room is there?  Distance from the exact value to the")
    print("nearest RNE decision boundary, in mantissa-code units:")
    exact = 128.0 * (target(fs) - 1.0)
    slack = 0.5 - np.abs(exact - refs)
    order = np.argsort(slack)
    print(f"    tightest input : {slack[order[0]]:.3e} codes")
    print(f"    10th tightest  : {slack[order[9]]:.3e} codes")
    print(f"    median         : {np.median(slack):.4f} codes")
    print("    (a polynomial must land inside that band for EVERY input)")
    print()

    print("minimax vs LP-retuned, on the truncated f the hardware sees")
    print(" deg | minimax wrong | LP margin (codes) | verdict")
    print("-----+---------------+-------------------+---------------------------")
    solutions = {}
    for deg in range(2, 7):
        mm = remez(target, deg)[0]
        wrong = count_wrong(mm, fs, refs)
        coeffs, margin = max_margin(fs, refs, deg)
        if coeffs is None:
            print(f" {deg:3d} | {wrong:13d} | {'infeasible':>17} | LP failed")
            continue
        solutions[deg] = (coeffs, margin)
        if margin > 0:
            verdict = f"FEASIBLE, {count_wrong(coeffs, fs, refs)} wrong"
        else:
            verdict = "INFEASIBLE at any coeffs"
        print(f" {deg:3d} | {wrong:13d} | {margin:17.4e} | {verdict}")

    print()
    print("A positive margin means SOME polynomial of that degree classifies all")
    print("inputs correctly.  A negative margin is a PROOF that no polynomial of")
    print("that degree can ever work, no matter how the coefficients are chosen.")
    print()

    for deg in (3, 4):
        if deg not in solutions:
            continue
        coeffs, margin = solutions[deg]
        print(f"--- degree {deg} LP solution ---")
        print(f"    margin : {margin:.4e} mantissa codes")
        for i, c in enumerate(coeffs):
            print(f"    c{i} = {c: .12f}")
        err = np.max(np.abs(np.polyval(coeffs[::-1], fs) - target(fs)))
        mm_err = np.max(np.abs(np.polyval(remez(target, deg)[0][::-1], fs) - target(fs)))
        print(f"    max |P - 2^(1-f)| : {err:.3e}   (minimax would be {mm_err:.3e})")
        print(f"    wrong codes       : {count_wrong(coeffs, fs, refs)}")
        print()


if __name__ == "__main__":
    main()
