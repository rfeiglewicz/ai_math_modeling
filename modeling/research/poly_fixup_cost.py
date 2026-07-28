#!/usr/bin/env python3
"""What would a fixup for the degree-3 polynomial actually cost?

poly_lp_retune.py proved that NO degree-3 polynomial classifies all 1952
distinct f values correctly (LP margin -7.6e-3 codes), so if degree 3 is to be
used the residual errors must be patched in hardware.

This script measures what such a patch would have to look like:

  1. structure of the residual errors (how many, which direction, clustered?),
  2. cost of exact-match detection (a CAM over f),
  3. cost of range-based detection (comparators over f intervals),
  4. the alternative: keep the cut ROM and let the polynomial only replace the
     candidate ROM, which turns the "fixup" into the single comparison the
     ladder already performs.

Option 4 is the interesting one: a polynomial that is merely within +/-1 of the
correct code is enough, because one cut comparison then resolves the ambiguity
EXACTLY.  That converts an unbounded correctness problem into a bounded one.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from poly_lp_retune import build_points, max_margin  # noqa: E402
from poly_minimax_study import remez, target  # noqa: E402

FRAC_BITS = 16
CUT_ROM_BITS = 1664      # cut ROM in the shipped cut-point ladder
CAND_ROM_BITS = 1024     # candidate ROM in the shipped cut-point ladder
TAIL_ROM_BITS = 77


def codes_of(coeffs, fs):
    return np.floor(128.0 * (np.polyval(coeffs[::-1], fs) - 1.0) + 0.5)


def analyse_errors(name, coeffs, fs, refs):
    codes = codes_of(coeffs, fs)
    bad = np.nonzero(codes != refs)[0]
    print(f"--- {name}: {len(bad)} residual errors ---")
    if len(bad) == 0:
        print("    nothing to patch")
        return bad

    delta = (refs[bad] - codes[bad]).astype(int)
    uniq, cnt = np.unique(delta, return_counts=True)
    print("    correction needed :", dict(zip(uniq.tolist(), cnt.tolist())))

    fq = (fs[bad] * (1 << FRAC_BITS)).astype(int)
    print(f"    distinct f values : {len(np.unique(fq))}")
    print(f"    f range           : {fq.min()} .. {fq.max()} (of 0..65535)")

    # Are the bad f values isolated, or do they form usable intervals?
    gaps = np.diff(np.sort(fq))
    runs = 1 + int(np.count_nonzero(gaps > 1))
    print(f"    contiguous runs   : {runs}  -> range detection needs {runs} window(s)")

    # Do neighbouring GOOD values sit right next to bad ones?
    allq = (fs * (1 << FRAC_BITS)).astype(int)
    badset = set(fq.tolist())
    touching = sum(
        1 for v in fq if (v - 1) in set(allq.tolist()) - badset or (v + 1) in set(allq.tolist()) - badset
    )
    print(f"    bad values adjacent to good ones : {touching} of {len(fq)}")
    print("    -> ranges cannot be widened; detection must be exact-match")
    return bad


def bracket_check(coeffs, fs, refs, label):
    """Is the polynomial always within +/-1 of the correct code?

    That is the ONLY property needed if a cut comparison follows, because the
    comparison then picks between base and base+1 using the exact boundary.
    """
    vals = np.polyval(coeffs[::-1], fs)
    base = np.floor(128.0 * (vals - 1.0))          # truncate, not round
    ok = np.all((refs == base) | (refs == base + 1))
    worst = np.max(np.abs(128.0 * (vals - 1.0) - refs))
    print(f"    {label:28} max |P-code - ref| = {worst:6.3f} codes -> "
          f"{'BRACKETS' if ok else 'does NOT bracket'}")
    return ok


def main():
    fs, refs = build_points(FRAC_BITS)
    print(f"distinct truncated f values : {len(fs)}\n")

    mm3 = remez(target, 3)[0]
    lp3, margin3 = max_margin(fs, refs, 3)
    print(f"degree-3 LP margin = {margin3:+.4e} codes -> provably infeasible\n")

    analyse_errors("degree 3 minimax", mm3, fs, refs)
    print()
    bad_lp = analyse_errors("degree 3 LP-retuned", lp3, fs, refs)
    print()

    n_bad = len(bad_lp)
    print("=== cost of patching those errors ===")
    cam = n_bad * (FRAC_BITS + 1)
    print(f" exact-match CAM   : {n_bad} entries x ({FRAC_BITS} b key + 1 b sign)"
          f" = {cam} b + {n_bad} comparators of {FRAC_BITS} b")
    print(f" the {FRAC_BITS}-b comparators alone are ~{n_bad * FRAC_BITS // 2} LUT6")
    print(" and they sit in series with the Horner chain, on the critical path")
    print()

    print("=== the alternative: polynomial replaces the CANDIDATE ROM only ===")
    print("A polynomial does not have to be exact if the cut ROM stays: it only")
    print("has to bracket the answer, so the existing single comparison fixes it.")
    print()
    for deg in (1, 2, 3):
        coeffs = remez(target, deg)[0]
        bracket_check(coeffs, fs, refs, f"minimax degree {deg}")
    print()

    print("=== resulting payload ===")
    full = CUT_ROM_BITS + CAND_ROM_BITS + TAIL_ROM_BITS
    print(f" shipped cut-point ladder      : {full} b, 0 variable multipliers")
    for deg in (2, 3):
        coeffs = remez(target, deg)[0]
        if bracket_check(coeffs, fs, refs, f"(check deg {deg})"):
            bits = CUT_ROM_BITS + TAIL_ROM_BITS + 24 * (deg + 1)
            print(f" cut ROM + degree-{deg} candidate : {bits} b, "
                  f"{deg} variable multipliers  "
                  f"(saves {full - bits} b, costs {deg} multipliers)")
    print()
    print("~950 saved bits of distributed ROM are worth ~15 LUT6; two variable")
    print("multipliers are worth ~2 DSP48.  The trade goes the wrong way.")


if __name__ == "__main__":
    main()
