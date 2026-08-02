#!/usr/bin/env python3
"""
Builds docs/pipe_target_comparison.md from build/vivado_compare/pipe_target_sweep.csv.

Run:  make pipe_target_table   (after: make sweep_pipe_targets)
"""
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV = ROOT / "build/vivado_compare/pipe_target_sweep.csv"
RES = ROOT / "build/vivado_compare/resources.csv"
OUT = ROOT / "docs/pipe_target_comparison.md"


def num(s):
    f = float(s)
    return int(f) if f == int(f) else f


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV}; run: make sweep_pipe_targets")

    rows = list(csv.DictReader(CSV.open()))
    targets = sorted({int(r["pipe_target"]) for r in rows})

    avail = {}
    if RES.exists():
        for r in csv.DictReader(RES.open()):
            if r["core"] == "AVAILABLE":
                avail = r

    L = []
    A = L.append
    A("# Koszt dopelniania potoku (PIPE_TARGET)")
    A("")
    A("Wygenerowane przez `make pipe_target_table`. Nie edytowac recznie.")
    A("")
    A("Uklad `xc7a200tfbg484-1`, Vivado 2025.2, ograniczenie zegara 2.0 ns.")
    A("Fmax po samej syntezie, bez implementacji.")
    A("")
    A("**PIPE_TARGET tylko dopelnia w gore.** Rdzen glebszy niz target zostaje")
    A("na swojej naturalnej glebokosci - nie da sie go skrocic. Takie wiersze sa")
    A("oznaczone `za plytki target` i powtarzaja wynik z natywnej glebokosci.")
    A("")

    for t in targets:
        sub = [r for r in rows if int(r["pipe_target"]) == t]
        A(f"## PIPE_TARGET = {t}")
        A("")
        A("| rdzen | stopnie | pad | LUT | CARRY4 | FF | SRL | DSP | BRAM36 | Fmax [MHz] | poziomy | uwaga |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in sub:
            A("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["core"], r["stages"], r["pad"], num(r["lut_logic"]),
                num(r["carry"]), num(r["ff"]), num(r["srl"]), num(r["dsp"]),
                num(r["bram36"]), r["fmax_mhz"], r["levels"],
                r["note"] or ""))
        A("")

    # Ile realnie kosztuje dopelnienie: roznica wzgledem najplytszego targetu,
    # liczona tylko tam gdzie dopelnienie faktycznie nastapilo.
    A("## Przyrost wzgledem naturalnej glebokosci")
    A("")
    A("Liczone tylko dla rdzeni, ktore faktycznie zostaly dopelnione.")
    A("")
    base = {}
    for r in rows:
        c = r["core"]
        if int(r["pad"]) == 0 and c not in base:
            base[c] = r
    A("| rdzen | target | pad | dLUT | dFF | dSRL | dFmax [MHz] |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if int(r["pad"]) == 0:
            continue
        b = base.get(r["core"])
        if not b:
            continue
        A("| {} | {} | +{} | {:+d} | {:+d} | {:+d} | {:+.1f} |".format(
            r["core"], r["pipe_target"], r["pad"],
            int(num(r["lut_logic"])) - int(num(b["lut_logic"])),
            int(num(r["ff"])) - int(num(b["ff"])),
            int(num(r["srl"])) - int(num(b["srl"])),
            float(r["fmax_mhz"]) - float(b["fmax_mhz"])))
    A("")

    if avail:
        A("## Dostepne w ukladzie")
        A("")
        A("| LUT | CARRY4 | FF | LUT-as-mem | DSP48E1 | BRAM36 |")
        A("|---:|---:|---:|---:|---:|---:|")
        A("| {} | {} | {} | {} | {} | {} |".format(
            num(avail["lut_logic"]), num(avail["carry"]), num(avail["ff"]),
            num(avail["srl"]), num(avail["dsp"]), num(avail["bram36"])))
        A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(rows)} rows, targets {targets})")


if __name__ == "__main__":
    main()
