#!/usr/bin/env python3
"""Degree 3 + fixup, or just degree 4?  Fixed-point verification.

poly_lp_retune.py proved degree 3 can never classify all inputs correctly, so
a fixup is MANDATORY, not optional.  poly_fixup_cost.py priced that fixup at
~20 exact-match comparators.

One extra Horner stage is far cheaper than 20 comparators, so the honest
comparison is "degree 3 + fixup" against "degree 4, LP-retuned".  This script
finds the cheapest bit-exact fixed-point degree-4 configuration and prints the
final cost table against the shipped cut-point ladder.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from poly_fixed_point_cost import eval_fixed  # noqa: E402
from poly_lp_retune import build_points, max_margin  # noqa: E402

FRAC_BITS = 16
LADDER_BITS = 2765


def sweep(coeffs, fs_q, refs, frac_bits):
    """Cheapest (coef_bits, data_bits) that reproduces every mantissa code."""
    for data_bits in range(16, 40):
        data_frac = data_bits - 2
        if data_frac < 8:
            continue
        half = 1 << (data_frac - 8)
        for coef_bits in range(12, 34):
            coef_frac = coef_bits - 2
            cq = [int(round(c * (1 << coef_frac))) for c in coeffs]
            ok = True
            for f_q, ref in zip(fs_q, refs):
                p = eval_fixed(cq, f_q, coef_frac, data_frac, frac_bits)
                code = ((p - (1 << data_frac)) + half) >> (data_frac - 7)
                if code != ref:
                    ok = False
                    break
            if ok:
                return coef_bits, data_bits
    return None


def main():
    fs, refs = build_points(FRAC_BITS)
    fs_q = [int(f * (1 << FRAC_BITS)) for f in fs]
    refs_i = [int(r) for r in refs]

    print(f"distinct truncated f values : {len(fs)}   (f width {FRAC_BITS} b)\n")

    print("=== degree 4, LP-retuned, in fixed point ===")
    coeffs, margin = max_margin(fs, refs, 4)
    print(f"    LP margin : {margin:.4e} mantissa codes")
    found = sweep(coeffs, fs_q, refs_i, FRAC_BITS)
    if found:
        coef_bits, data_bits = found
        rom = coef_bits * 5
        print(f"    cheapest bit-exact : {coef_bits} b coefficients, "
              f"{data_bits} b datapath")
        print(f"    coefficient storage: {rom} b, 4 variable multipliers, "
              f"f stays {FRAC_BITS} b")
    else:
        rom = None
        print("    no bit-exact configuration found up to 33/39 bits")
    print()

    print("=== final comparison ===")
    print(" variant                          | tables | var mult | extra logic")
    print("----------------------------------+--------+----------+---------------")
    print(f" cut-point ladder (shipped)       | {LADDER_BITS:5d} b |        0 | "
          f"1 comparator")
    print(f" degree 3 + exact-match fixup     |   360 b |        3 | "
          f"20 x 16-b comparators")
    if rom:
        print(f" degree 4, LP-retuned             |   {rom:3d} b |        4 | "
              f"none")
    print()
    print("One extra Horner stage replaces 20 comparators AND 340 b of CAM, so")
    print("patching degree 3 is never the cheaper option.")
    print()
    print("Against the ladder the trade is unchanged: ~2.6 kb of distributed ROM")
    print("(~45 LUT6) versus 4 variable-operand multipliers (~4 DSP48 or ~2-3 kLUT),")
    print("plus RNE rounding logic the ladder does not need, plus a 4-deep serial")
    print("multiply chain instead of one comparator.")


if __name__ == "__main__":
    main()
