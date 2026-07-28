#!/usr/bin/env python3
"""Pick a DSP48-friendly degree-4 configuration for BF16 exp(x).

Goal: a back end that MAPS CLEANLY ONTO DSP SLICES instead of avoiding them.

A Xilinx DSP48E1 multiplies a 25-bit A port by an 18-bit B port and can add the
C port / PCIN cascade to the product for free.  A Horner step

    acc <- acc * f + c_k

is therefore exactly one DSP slice when

    acc fits in 25 bits (signed)   ->  A port
    f   fits in 18 bits (unsigned) ->  B port
    c_k is added through C / PCIN  ->  free

So the natural design point is f = 18 bits, accumulator = 25 bits, and a degree-4
polynomial becomes a 4-slice DSP cascade.

This script reports, for several f widths, how much margin the LP has, so the
C++ generator can be given a configuration it can hit reliably with a plain
integer search instead of a full LP solver.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from poly_lp_retune import build_points, max_margin  # noqa: E402
from poly_minimax_study import remez, target  # noqa: E402


def truncation_slack(fs, refs):
    """Distance from 2^(1-f_truncated) to the nearest RNE boundary, in codes.

    Negative means truncating f already pushed the exact value across a
    decision boundary, so the polynomial must deliberately compensate.
    """
    exact = 128.0 * (target(fs) - 1.0)
    return 0.5 - np.abs(exact - refs)


def main():
    print("f width | distinct f | worst trunc slack | LP margin deg 4 | minimax deg 4")
    print("--------+------------+-------------------+-----------------+--------------")
    for bits in (16, 17, 18, 19, 20, 22):
        fs, refs = build_points(bits)
        slack = truncation_slack(fs, refs)
        _, margin = max_margin(fs, refs, 4)
        mm = remez(target, 4)[0]
        codes = np.floor(128.0 * (np.polyval(mm[::-1], fs) - 1.0) + 0.5)
        wrong = int(np.count_nonzero(codes != refs))
        print(
            f" {bits:6d} | {len(fs):10d} | {slack.min():17.3e} | "
            f"{margin:15.3e} | {wrong:5d} wrong"
        )

    print()
    print("A positive 'worst trunc slack' means truncation alone never crosses a")
    print("boundary, so an ordinary fit has room to work with and the generator")
    print("does not need a full LP solver.")
    print()

    for bits in (18, 20):
        fs, refs = build_points(bits)
        coeffs, margin = max_margin(fs, refs, 4)
        print(f"=== f = {bits} bits: LP margin {margin:.4e} codes ===")
        for i, c in enumerate(coeffs):
            print(f"    c{i} = {c: .12f}")

        # How coarse is an 18-bit coefficient grid compared to that margin?
        lsb = 2.0 ** -(18 - 2)
        print(f"    18-b coefficient LSB = {lsb:.3e} value units "
              f"= {lsb * 128:.4f} codes")
        print(f"    margin in value units = {margin / 128:.3e}")
        print()


if __name__ == "__main__":
    main()
