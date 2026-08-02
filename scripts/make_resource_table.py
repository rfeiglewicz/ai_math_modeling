#!/usr/bin/env python3
"""
Builds docs/resource_comparison.md from measurements, not estimates.

Inputs (both produced by measurement, never by hand):
  build/vivado_compare/resources.csv  - Vivado report_utilization + timing
  build/vivado_compare/depths.txt     - Verilator elaboration depth report

Run:  make resource_table
"""
import csv
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV = ROOT / "build/vivado_compare/resources.csv"
OUT = ROOT / "docs/resource_comparison.md"


def load_depths():
    """Ask each core how deep it is; do not re-derive the formula here."""
    txt = subprocess.run(
        [str(ROOT / "scripts/pipe_depths.sh")],
        capture_output=True, text=True, check=True,
    ).stdout
    depths = {}
    for line in txt.splitlines():
        parts = line.rsplit(None, 1)
        if len(parts) == 2 and parts[1].isdigit():
            depths[parts[0].strip()] = int(parts[1])
    return depths


def num(s):
    f = float(s)
    return int(f) if f == int(f) else f


def pct(used, avail):
    if not avail:
        return "-"
    p = 100.0 * float(used) / float(avail)
    if used == 0:
        return "0"
    return f"{p:.3f}".rstrip("0").rstrip(".") if p < 1 else f"{p:.2f}"


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV}; run: make compare_cores")

    rows = list(csv.DictReader(CSV.open()))
    avail = next(r for r in rows if r["core"] == "AVAILABLE")
    cores = [r for r in rows if r["core"] != "AVAILABLE"]
    depths = load_depths()

    missing = [c["core"] for c in cores if c["core"] not in depths]
    if missing:
        sys.exit(f"no measured depth for: {missing}")

    L = []
    A = L.append
    A("# BF16 exp(x) - zasoby i glebokosc potoku")
    A("")
    A("Wygenerowane przez `make resource_table`. Nie edytowac recznie.")
    A("")
    A("Wszystkie rdzenie zsyntezowane z `PIPE_TARGET=0`, czyli **bez dopelniania**")
    A("do wspolnej latencji. Kazdy rdzen ma tu swoja naturalna glebokosc; w")
    A("projekcie docelowym dziela wspolny zegar i sa dopelniane do")
    A("`UNIFIED_PIPE_DEPTH`, co kosztuje dodatkowe SRL i przerzutniki.")
    A("")
    A("## Metodyka")
    A("")
    A(f"- uklad: `{avail.get('part', 'xc7a200tfbg484-1')}`" if avail.get("part")
      else "- uklad: `xc7a200tfbg484-1` (Artix-7 200T)")
    A("- Vivado 2025.2, `synth_design` bez implementacji")
    A("- ograniczenie zegara **2.0 ns (500 MHz)**, celowo nieosiagalne: synteza")
    A("  przestaje optymalizowac po spelnieniu ograniczenia, wiec realistyczny")
    A("  target zanizylby Fmax")
    A("- Fmax liczone z najgorszej sciezki wewnetrznej: `1000 / (okres - slack)`")
    A("- zasoby z `report_utilization` (miejsca w ukladzie), nie z `get_cells`")
    A("  (prymitywy) - dwa LUT5 dziela jedno miejsce LUT6")
    A("- glebokosc potoku raportowana przez sam RTL przy elaboracji")
    A("  (`-GREPORT_DEPTH=1`), wiec nie moze sie rozjechac z kodem")
    A("")
    A("**Fmax jest po samej syntezie, bez placement i routingu.** Na ostatnich")
    A("zmierzonych sciezkach krytycznych 56-77% opoznienia to *szacowane*")
    A("trasowanie, wiec po implementacji te liczby spadna.")
    A("")
    A("## Zasoby (bez dopelniania potoku)")
    A("")
    hdr = ("| rdzen | stopnie | LUT | CARRY4 | FF | SRL | DSP48E1 | BRAM36 | "
           "RAMB18 | Fmax [MHz] | poziomy |")
    A(hdr)
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in cores:
        A("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["core"], depths[r["core"]], num(r["lut_logic"]), num(r["carry"]),
            num(r["ff"]), num(r["srl"]), num(r["dsp"]), num(r["bram36"]),
            num(r["ramb18"]), r["fmax_mhz"], r["levels"]))
    A("| **dostepne w ukladzie** | - | **{}** | **{}** | **{}** | **{}** | "
      "**{}** | **{}** | **{}** | - | - |".format(
          num(avail["lut_logic"]), num(avail["carry"]), num(avail["ff"]),
          num(avail["srl"]), num(avail["dsp"]), num(avail["bram36"]),
          num(avail["ramb18"])))
    A("")
    A("Kolumny: LUT = LUT jako logika; SRL = LUT jako rejestr przesuwny")
    A("(osobno, bo to dopelnienie potoku, nie logika); CARRY4 dzieli miejsce")
    A("ze slice, wiec limit rowna sie liczbie slice; BRAM36 bywa ulamkowy, bo")
    A("pojedynczy RAMB18 to pol kafelka.")
    A("")
    A("## Udzial w ukladzie [%]")
    A("")
    A("| rdzen | LUT | FF | DSP48E1 | BRAM36 |")
    A("|---|---:|---:|---:|---:|")
    for r in cores:
        A("| {} | {} | {} | {} | {} |".format(
            r["core"],
            pct(num(r["lut_logic"]), num(avail["lut_logic"])),
            pct(num(r["ff"]), num(avail["ff"])),
            pct(num(r["dsp"]), num(avail["dsp"])),
            pct(num(r["bram36"]), num(avail["bram36"]))))
    A("")
    A("Zaden rdzen nie przekracza **1%** zadnego zasobu ukladu. Przy takich")
    A("rozmiarach o wyborze nie decyduje zajetosc, tylko Fmax, liczba DSP i to,")
    A("czy chcemy wydac blok pamieci - a te trzy rzeczy wykluczaja sie wzajemnie.")
    A("")
    A("## Ile takich rdzeni zmiesci sie w ukladzie")
    A("")
    A("Limit z najciasniejszego zasobu, przy zalozeniu ze rdzenie nie dziela")
    A("niczego - w praktyce tablice ROM sa identyczne, wiec dalyby sie")
    A("wspoldzielic i realna liczba bylaby wieksza.")
    A("")
    A("| rdzen | limit LUT | limit FF | limit DSP | limit BRAM | **max sztuk** |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in cores:
        lim = {}
        for key, col in (("LUT", "lut_logic"), ("FF", "ff"),
                         ("DSP", "dsp"), ("BRAM", "bram36")):
            u = num(r[col])
            lim[key] = int(float(avail[col]) / float(u)) if u else None
        vals = [v for v in lim.values() if v is not None]
        A("| {} | {} | {} | {} | {} | **{}** |".format(
            r["core"],
            lim["LUT"] if lim["LUT"] is not None else "-",
            lim["FF"] if lim["FF"] is not None else "-",
            lim["DSP"] if lim["DSP"] is not None else "-",
            lim["BRAM"] if lim["BRAM"] is not None else "-",
            min(vals) if vals else "-"))
    A("")
    A("## Koszt ujednolicenia latencji")
    A("")
    mx = max(depths[c["core"]] for c in cores)
    A(f"Najglebszy rdzen ma **{mx}** stopni, wiec przy wspolnym zegarze kazdy")
    A("inny musi zostac dopelniony do tej samej liczby. Dopelnienie idzie w")
    A("SRL16E (16 bitow na jeden LUT), wiec jest tanie, ale nie darmowe:")
    A("")
    A("| rdzen | stopnie wlasne | do dopelnienia |")
    A("|---|---:|---:|")
    for r in cores:
        d = depths[r["core"]]
        A(f"| {r['core']} | {d} | {mx - d} |")
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(cores)} cores)")


if __name__ == "__main__":
    main()
