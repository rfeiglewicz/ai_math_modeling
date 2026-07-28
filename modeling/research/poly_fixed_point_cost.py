#!/usr/bin/env python3
"""Fixed-point cost study for the minimax/Horner back end of BF16 exp(x).

poly_minimax_study.py showed that degree 3 cannot reach 0.5 ULP and that
degree 5 is the first degree with real headroom in EXACT arithmetic.  Real
hardware adds two more error sources:

  * coefficients must be quantized to a finite number of bits,
  * every Horner product must be truncated to a finite datapath width.

This script sweeps (coefficient bits, datapath bits) for degrees 4..6 and
reports the cheapest configuration that is still bit-exact on all 1967 inputs,
plus the resulting multiplier/ROM cost next to the cut-point ladder.
"""

import math

import numpy as np

from poly_minimax_study import (
    ladder_inputs,
    reference_code,
    remez,
    target,
)

# Cut-point ladder reference numbers (measured, see gen_bf16_expe_cut_tables).
LADDER_BITS = 2765
LADDER_VAR_MULTS = 0      # the only multiply is by a constant
LADDER_CONST_MULTS = 1    # 8 x 24 CSD adder tree
LADDER_COMPARES = 1


def rescale(value, from_frac, to_frac):
    """Move a fixed-point value between two fractional scalings."""
    if to_frac >= from_frac:
        return value << (to_frac - from_frac)
    return value >> (from_frac - to_frac)


def eval_fixed(coeffs_q, f_q, coef_frac, data_frac, frac_bits):
    """Horner in integer arithmetic, mirroring what RTL would do.

    coeffs_q  : coefficients scaled by 2**coef_frac, ascending order
    f_q       : f scaled by 2**frac_bits (this is the datapath value)
    returns   : P(f) scaled by 2**data_frac
    """
    acc = rescale(coeffs_q[-1], coef_frac, data_frac)
    for c in reversed(coeffs_q[:-1]):
        # acc * f  ->  scale data_frac + frac_bits, truncate back to data_frac
        acc = (acc * f_q) >> frac_bits
        acc += rescale(c, coef_frac, data_frac)
    return acc


_REMEZ_CACHE = {}


def minimax_coeffs(deg):
    """Remez coefficients, computed once per degree."""
    if deg not in _REMEZ_CACHE:
        _REMEZ_CACHE[deg] = remez(target, deg)[0]
    return _REMEZ_CACHE[deg]


def try_config(deg, coef_bits, data_bits, fs_q, refs, frac_bits):
    """Return number of wrong mantissa codes for one (coef, data) width."""
    coeffs = minimax_coeffs(deg)

    # Coefficients live in [-2, 2], so one integer bit plus sign is enough.
    coef_frac = coef_bits - 2
    coeffs_q = [int(round(c * (1 << coef_frac))) for c in coeffs]

    data_frac = data_bits - 2          # P(f) in (1, 2], one integer bit
    half = 1 << (data_frac - 8)        # 0.5 ULP of the 7-bit mantissa

    wrong = 0
    for f_q, ref in zip(fs_q, refs):
        p = eval_fixed(coeffs_q, f_q, coef_frac, data_frac, frac_bits)
        # mantissa code = round(128 * (P - 1)) = round((P - 1) << 7)
        code = ((p - (1 << data_frac)) + half) >> (data_frac - 7)
        if code != ref:
            wrong += 1
    return wrong


def main():
    pts = ladder_inputs()
    fs = [f for _, f in pts]
    refs = [reference_code(f) for f in fs]

    print(f"test vectors : {len(pts)}")
    print()
    print("Why the ladder gets away with 16-bit f but a polynomial does not:")
    print("  the ladder MOVES its decision boundaries between the f values that")
    print("  actually occur, so truncating f is harmless.  A polynomial is a")
    print("  smooth function of f, so every bit dropped from f shows up directly")
    print("  in P(f).  Sweeping the f width makes the difference explicit.")
    print()

    print(" f bits | deg 3 | deg 4 | deg 5 | deg 6   (wrong mantissa codes)")
    print("--------+-------+-------+-------+--------")
    for frac_bits in (16, 18, 20, 22, 23, 24, 26):
        fs_q = [int(f * (1 << frac_bits)) for f in fs]
        row = []
        for deg in range(3, 7):
            best = min(
                try_config(deg, coef_bits, 32, fs_q, refs, frac_bits)
                for coef_bits in (20, 24, 28)
            )
            row.append(best)
        print(
            f" {frac_bits:6d} | {row[0]:5d} | {row[1]:5d} | "
            f"{row[2]:5d} | {row[3]:5d}"
        )

    print()
    print("cheapest bit-exact fixed-point configuration per degree")
    print(" deg | f bits | coef bits | data bits | var mults | coef ROM")
    print("-----+--------+-----------+-----------+-----------+----------")

    summary = {}
    for deg in range(3, 7):
        found = None
        for frac_bits in range(16, 31):
            fs_q = [int(f * (1 << frac_bits)) for f in fs]
            for data_bits in range(16, 37):
                for coef_bits in range(12, 33):
                    if try_config(deg, coef_bits, data_bits, fs_q, refs, frac_bits) == 0:
                        found = (frac_bits, coef_bits, data_bits)
                        break
                if found:
                    break
            if found:
                break
        if found:
            frac_bits, coef_bits, data_bits = found
            rom = coef_bits * (deg + 1)
            summary[deg] = (frac_bits, coef_bits, data_bits, deg, rom)
            print(
                f" {deg:3d} | {frac_bits:6d} | {coef_bits:9d} | {data_bits:9d} | "
                f"{deg:9d} | {rom:6d} b"
            )
        else:
            print(f" {deg:3d} |   --   |    --     |    --     | {deg:9d} |   --")

    print()
    print("cost comparison")
    print(" method                | f bits | table bits | const mult | var mult | cmp")
    print("----------------------+--------+------------+------------+----------+-----")
    print(
        f" cut-point ladder     | {16:6d} | {LADDER_BITS:10d} | "
        f"{LADDER_CONST_MULTS:10d} | {LADDER_VAR_MULTS:8d} | {LADDER_COMPARES:3d}"
    )
    for deg, (frac_bits, coef_bits, data_bits, mults, rom) in sorted(summary.items()):
        print(
            f" minimax Horner deg {deg} | {frac_bits:6d} | {rom:10d} | "
            f"{1:10d} | {mults:8d} | {0:3d}"
        )

    print()
    print("note: the ladder's single multiply has a CONSTANT operand (CSD adder")
    print("      tree); every Horner multiply has TWO variable operands and maps")
    print("      to a real multiplier / DSP slice.  A wider f also widens the")
    print("      constant multiplier and the alignment barrel shifter.")


if __name__ == "__main__":
    main()
