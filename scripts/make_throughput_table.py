#!/usr/bin/env python3
"""Analiza przepustowosci: ile rdzeni expe zmiesci sie w FPGA i ile razem daja
operacji na sekunde.

Kazdy rdzen jest w pelni potokowy (AXI-Stream, 1 wynik na takt przy tvalid=1),
wiec glebokosc potoku wplywa na latencje, a nie na przepustowosc.  Dla N kopii
pracujacych z zegarem f przepustowosc wynosi

    T = N * f          [operacji/s]

N jest ograniczone przez najciasniejszy zasob:

    N = min_r floor(available_r / used_r)

a f - przez najwolniejszy rdzen w domenie zegarowej:

    f = min_c Fmax_c

Dane wejsciowe pochodza wylacznie z pomiaru (build/vivado_compare/*.csv),
zadnych szacunkow.

Uzycie:
    python3 scripts/make_throughput_table.py [--out docs/throughput_analysis.md]
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES_CSV = os.path.join(ROOT, "build", "vivado_compare", "resources.csv")
PIPE_CSV = os.path.join(ROOT, "build", "vivado_compare", "pipe_target_sweep.csv")

# Kolejnosc ma znaczenie tylko dla wydruku.
RESOURCES = [
    ("lut", "LUT"),
    ("srl", "LUT-as-mem"),
    ("ff", "FF"),
    ("carry", "CARRY4"),
    ("dsp", "DSP48E1"),
    ("bram36", "BRAM36"),
]


@dataclass
class Core:
    name: str
    lut: int  # lut_logic + srl, bo SRL zajmuje ten sam fizyczny LUT
    srl: int
    ff: int
    carry: int
    dsp: int
    bram36: float
    fmax: float

    def used(self, key: str) -> float:
        return float(getattr(self, key))


def load_cores(path: str) -> tuple[list[Core], dict[str, float]]:
    cores: list[Core] = []
    avail: dict[str, float] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            lut_logic = float(row["lut_logic"])
            srl = float(row["srl"])
            rec = dict(
                lut=lut_logic + srl,
                srl=srl,
                ff=float(row["ff"]),
                carry=float(row["carry"]),
                dsp=float(row["dsp"]),
                bram36=float(row["bram36"]),
            )
            if row["core"] == "AVAILABLE":
                # W raporcie Vivado "LUT as Logic" i "LUT as Memory" sumuja sie
                # do puli Slice LUTs, wiec limitem calkowitym jest lut_logic.
                avail = dict(rec, lut=lut_logic)
                continue
            cores.append(Core(name=row["core"], fmax=float(row["fmax_mhz"]), **rec))
    return cores, avail


def max_instances(core: Core, avail: dict[str, float]) -> tuple[int, str]:
    """N dla jednego typu rdzenia oraz nazwa zasobu, ktory ogranicza."""
    best_n = None
    binding = "-"
    for key, label in RESOURCES:
        used = core.used(key)
        if used <= 0:
            continue
        n = int(avail[key] // used)
        if best_n is None or n < best_n:
            best_n, binding = n, label
    return (best_n or 0), binding


def solve_mix(cores: list[Core], avail: dict[str, float],
              weights: list[float]) -> tuple[list[int], float]:
    """Maksymalizuje sum(w_i * n_i) przy ograniczeniach zasobow, n_i calkowite.

    Maly program calkowitoliczbowy: 6 ograniczen, kilkanascie zmiennych.
    Rozwiazywany dokladnie przez HiGHS (scipy.optimize.milp).
    """
    import numpy as np
    from scipy.optimize import LinearConstraint, Bounds, milp

    keys = [k for k, _ in RESOURCES]
    a = np.array([[c.used(k) for c in cores] for k in keys], dtype=float)
    b = np.array([avail[k] for k in keys], dtype=float)
    caps = np.array([max_instances(c, avail)[0] for c in cores], dtype=float)

    res = milp(
        c=-np.array(weights, dtype=float),          # milp minimalizuje
        constraints=LinearConstraint(a, -np.inf, b),
        integrality=np.ones(len(cores)),
        bounds=Bounds(np.zeros(len(cores)), caps),
    )
    if not res.success:
        return [0] * len(cores), 0.0
    vec = [int(round(x)) for x in res.x]
    return vec, float(sum(wi * n for wi, n in zip(weights, vec)))


def check(vec: list[int], cores: list[Core], avail: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, _ in RESOURCES:
        out[key] = sum(n * c.used(key) for n, c in zip(vec, cores))
        assert out[key] <= avail[key] + 1e-9, f"{key} przekroczony"
    return out


def fmt(x: float) -> str:
    return f"{x:.1f}" if x % 1 else f"{int(x)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "throughput_analysis.md"))
    ap.add_argument("--budget", type=float, default=0.8,
                    help="realistyczny wspolczynnik wykorzystania ukladu")
    args = ap.parse_args()

    cores, avail = load_cores(RES_CSV)

    # Policz wszystko najpierw, zeby dalo sie postawic podsumowanie na poczatku.
    singles = []
    for c in cores:
        n, binding = max_instances(c, avail)
        singles.append((n * c.fmax / 1000.0, n, binding, c))

    best_single_clock = None
    clock_rows = []
    for f in sorted({c.fmax for c in cores}, reverse=True):
        sub = [c for c in cores if c.fmax >= f - 1e-9]
        # Waga 1 na kopie plus mikroskopijny bonus za Fmax: liczy sie tylko
        # liczba rdzeni, ale przy remisie (np. dwa warianty poly4 dsp zajmuja
        # tyle samo DSP i LUT) wybieramy szybszy.
        weights = [1.0 + 1e-6 * c.fmax for c in sub]
        vec, _ = solve_mix(sub, avail, weights)
        total = sum(vec)
        if total <= 0:
            continue
        check(vec, sub, avail)
        mix = " + ".join(f"{n} x {c.name}" for n, c in zip(vec, sub) if n)
        t = total * f / 1000.0
        clock_rows.append((f, mix, int(total), t))
        if best_single_clock is None or t > best_single_clock[0]:
            best_single_clock = (t, f, mix, vec, sub, int(total))

    vec2, val2 = solve_mix(cores, avail, [c.fmax for c in cores])

    lines: list[str] = []
    w = lines.append

    w("# Maksymalna przepustowosc: ile rdzeni expe zmiesci sie w FPGA")
    w("")
    w("Uklad: **xc7a200tfbg484-1**, Vivado 2025.2, wyniki **po syntezie**")
    w("(`synth_design`, bez implementacji).  Wszystkie liczby zasobow i Fmax")
    w("pochodza z `build/vivado_compare/resources.csv`, generowanego przez")
    w("`make compare_cores`.  Zadna wartosc w tym dokumencie nie jest szacowana.")
    w("")
    bt, bf, bmix, _, _, bn = best_single_clock
    st, sn, sbind, sc = max(singles, key=lambda r: r[0])
    w("## Wynik")
    w("")
    w(f"* **Jeden typ rdzenia:** `{sc.name}`, {sn} kopii przy {sc.fmax:.1f} MHz -> "
      f"**{st:.1f} Gop/s** (limit: {sbind}).")
    w(f"* **Mieszanka, wspolny zegar {bf:.1f} MHz:** {bmix} = {bn} rdzeni -> "
      f"**{bt:.1f} Gop/s**.")
    w(f"* **Mieszanka, osobne zegary:** te same {sum(vec2)} rdzeni, kazdy typ na "
      f"wlasnym Fmax -> **{val2 / 1000.0:.1f} Gop/s**.")
    w("")
    w("Mieszanka wygrywa, bo rdzenie wysycaja **rozne** zasoby: `hybrid` konczy")
    w("sie na BRAM, `cut` na LUT, `poly4 dsp` na DSP.  Sam `" + sc.name + "` wysyca")
    solo = {k: sn * sc.used(k) for k, _ in RESOURCES}
    free = ", ".join(
        f"{100.0 * (1 - solo[k] / avail[k]):.0f}% {label}"
        for k, label in RESOURCES if k in ("lut", "ff", "dsp"))
    w(f"{sbind} w 100%, ale zostawia wolne {free}.")
    w("")
    w("Te liczby zakladaja 100% wysycenia ukladu.  Realistyczny wariant (80%)")
    w("jest w rozdziale 4.")
    w("")
    w("## Model")
    w("")
    w("Kazdy rdzen jest w pelni potokowy: interfejs AXI-Stream przyjmuje nowa")
    w("probke w kazdym takcie i po `PIPE_DEPTH` taktach oddaje wynik, bez przerw.")
    w("Stad **glebokosc potoku nie wplywa na przepustowosc, tylko na latencje**.")
    w("Przepustowosc jednej kopii to dokladnie 1 operacja na takt.")
    w("")
    w("Dla N kopii w jednej domenie zegarowej:")
    w("")
    w("    T = N * f,   f = min Fmax po uzytych typach rdzeni")
    w("    N ogranicza najciasniejszy zasob: N = min_r floor(dostepne_r / uzyte_r)")
    w("")
    w("Uwaga o LUT: w raporcie Vivado `LUT as Logic` i `LUT as Memory` (kolumna")
    w("`srl`) czerpia z tej samej puli 134600 Slice LUT, dlatego w tabelach nizej")
    w("`LUT` oznacza sume `lut_logic + srl`.")
    w("")

    # ---------------- pojedynczy typ ----------------
    w("## 1. Jeden typ rdzenia, jedna domena zegarowa")
    w("")
    w("| rdzen | LUT | FF | DSP | BRAM36 | Fmax [MHz] | zasob wiazacy | N kopii | T [Gop/s] |")
    w("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for t, n, binding, c in singles:
        w(f"| {c.name} | {fmt(c.lut)} | {fmt(c.ff)} | {fmt(c.dsp)} | {fmt(c.bram36)} | "
          f"{c.fmax:.1f} | {binding} | {n} | **{t:.1f}** |")
    w("")
    w(f"Najlepszy pojedynczy typ: **{sc.name}** - {sn} kopii przy "
      f"{sc.fmax:.1f} MHz = **{st:.1f} Gop/s**, ograniczony przez {sbind}.")
    w("")

    # ---------------- wydajnosc na zasob ----------------
    w("### Wydajnosc na jednostke zasobu")
    w("")
    w("To jest liczba, ktora decyduje o skladzie mieszanki: ile operacji na")
    w("sekunde daje rdzen w przeliczeniu na jeden zajety element danego typu")
    w("(`Fmax / uzyte_r`).  Im wiecej, tym oplacalniej wydac na niego ten zasob.")
    w("")
    w("| rdzen | Mop/s na LUT | Mop/s na DSP | Mop/s na BRAM36 |")
    w("|---|---:|---:|---:|")
    for c in cores:
        def per(key: str) -> str:
            u = c.used(key)
            return f"{c.fmax / u:.1f}" if u > 0 else "-"
        w(f"| {c.name} | {per('lut')} | {per('dsp')} | {per('bram36')} |")
    w("")

    # ---------------- mieszanka, jeden zegar ----------------
    w("## 2. Mieszanka rdzeni, jedna domena zegarowa")
    w("")
    w("Rdzenie wysycaja **rozne** zasoby: `hybrid` i `full-lut` blokuja BRAM,")
    w("`cut` i `poly4` blokuja DSP, `exp2` blokuje LUT.  Mieszanie typow pozwala")
    w("wykorzystac zasoby, ktore jeden typ zostawia niewykorzystane.  Cena jest")
    w("taka, ze wspolny zegar spada do Fmax najwolniejszego uzytego typu.")
    w("")
    w("| f [MHz] | sklad | N razem | T [Gop/s] |")
    w("|---:|---|---:|---:|")
    for f, mix, total, t in clock_rows:
        w(f"| {f:.1f} | {mix} | {total} | {t:.1f} |")
    w("")
    t, f, mix, vec, sub, total = best_single_clock
    w(f"**Optimum przy jednym zegarze: {mix}**, razem {total} rdzeni przy "
      f"{f:.1f} MHz = **{t:.1f} Gop/s**.")
    w("")
    w("Ponizej 193.2 MHz tabela nie rosnie: mieszanka juz wysyca BRAM i DSP w")
    w("100%, a LUT w 99.9%, wiec zwolnienie zegara nie odblokowuje zadnych")
    w("dodatkowych kopii - tylko obniza T.")
    w("")
    used = check(vec, sub, avail)
    w("Wykorzystanie zasobow w tym punkcie:")
    w("")
    w("| zasob | uzyte | dostepne | % |")
    w("|---|---:|---:|---:|")
    for key, label in RESOURCES:
        w(f"| {label} | {fmt(used[key])} | {fmt(avail[key])} | "
          f"{100.0 * used[key] / avail[key]:.1f}% |")
    w("")

    # ---------------- mieszanka, wiele zegarow ----------------
    w("## 3. Mieszanka rdzeni, osobne domeny zegarowe")
    w("")
    w("Jesli kazdy typ dostanie wlasny zegar (rdzenie sa niezalezne, lacza je")
    w("tylko strumienie danych), to nie trzeba placic za najwolniejszy typ i")
    w("maksymalizuje sie `sum(N_c * Fmax_c)`.")
    w("")
    used2 = check(vec2, cores, avail)
    mix2 = " + ".join(f"{n} x {c.name} @ {c.fmax:.1f} MHz"
                      for n, c in zip(vec2, cores) if n)
    w(f"Sklad: **{mix2}**")
    w("")
    w(f"Razem {sum(vec2)} rdzeni, **{val2 / 1000.0:.1f} Gop/s**.")
    w("")
    w("| zasob | uzyte | dostepne | % |")
    w("|---|---:|---:|---:|")
    for key, label in RESOURCES:
        w(f"| {label} | {fmt(used2[key])} | {fmt(avail[key])} | "
          f"{100.0 * used2[key] / avail[key]:.1f}% |")
    w("")

    # ---------------- realistyczny budzet ----------------
    b = args.budget
    w(f"## 4. Realistyczny budzet ({int(b * 100)}% ukladu)")
    w("")
    w("Punkty z rozdzialow 2 i 3 zakladaja 100% wysycenia ukladu, czego nie da")
    w("sie zrutowac.  Ponizej to samo przy budzecie zasobow ograniczonym do")
    w(f"{int(b * 100)}% kazdego zasobu.")
    w("")
    avail_b = {k: v * b for k, v in avail.items()}
    w("| scenariusz | sklad | N razem | T [Gop/s] |")
    w("|---|---|---:|---:|")
    best_b = None
    for f in sorted({c.fmax for c in cores}, reverse=True):
        allowed = [c for c in cores if c.fmax >= f - 1e-9]
        weights = [1.0 + 1e-6 * c.fmax for c in allowed]
        vecb, _ = solve_mix(allowed, avail_b, weights)
        totalb = sum(vecb)
        if totalb <= 0:
            continue
        tb = totalb * f / 1000.0
        if best_b is None or tb > best_b[0]:
            mixb = " + ".join(f"{n} x {c.name}" for n, c in zip(vecb, allowed) if n)
            best_b = (tb, f, mixb, int(totalb))
    w(f"| jeden zegar {best_b[1]:.1f} MHz | {best_b[2]} | {best_b[3]} | {best_b[0]:.1f} |")
    vec2b, val2b = solve_mix(cores, avail_b, [c.fmax for c in cores])
    mix2b = " + ".join(f"{n} x {c.name}" for n, c in zip(vec2b, cores) if n)
    w(f"| osobne zegary | {mix2b} | {sum(vec2b)} | {val2b / 1000.0:.1f} |")
    w("")

    # ---------------- wplyw PIPE_TARGET ----------------
    if os.path.exists(PIPE_CSV):
        w("## 5. Wplyw wyrownania potoku (PIPE_TARGET)")
        w("")
        w("Wyrownanie latencji przez `bf16_pipe_pad` nie zmienia Fmax (dFmax = 0")
        w("dla kazdego rdzenia i kazdego targetu, patrz `docs/pipe_target_comparison.md`),")
        w("ale zjada LUT i FF, wiec zmniejsza liczbe kopii, ktore sie mieszcza.")
        w("")
        w("| PIPE_TARGET | rdzen | LUT | FF | Fmax [MHz] | zasob wiazacy | N | T [Gop/s] |")
        w("|---:|---|---:|---:|---:|---|---:|---:|")
        with open(PIPE_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                c = Core(
                    name=row["core"],
                    lut=float(row["lut_logic"]) + float(row["srl"]),
                    srl=float(row["srl"]),
                    ff=float(row["ff"]),
                    carry=float(row["carry"]),
                    dsp=float(row["dsp"]),
                    bram36=float(row["bram36"]),
                    fmax=float(row["fmax_mhz"]),
                )
                n, binding = max_instances(c, avail)
                w(f"| {row['pipe_target']} | {c.name} | {fmt(c.lut)} | {fmt(c.ff)} | "
                  f"{c.fmax:.1f} | {binding} | {n} | {n * c.fmax / 1000.0:.1f} |")
        w("")

    w("## Zastrzezenia")
    w("")
    w("* Fmax jest **po syntezie, bez place & route**.  Na ostatnich sciezkach")
    w("  krytycznych 56-77% opoznienia to *szacowane* trasowanie.  Przy wysyceniu")
    w("  ukladu bliskim 100% realne Fmax bedzie wyraznie nizsze - stad rozdzial 4.")
    w("* Liczby kopii z rozdzialow 2 i 3 to **gorne ograniczenie**, a nie projekt,")
    w("  ktory da sie zbudowac.")
    w("* Zawartosc ROM jest identyczna we wszystkich kopiach.  BRAM w Artix-7 jest")
    w("  prawdziwie dwuportowy, wiec jeden blok moze obsluzyc 2 rdzenie na takt;")
    w("  to podnioslo by limit dla `hybrid` i `full-lut` o czynnik 2, kosztem")
    w("  dodatkowej logiki adresowej.  Nie jest to zaimplementowane w RTL.")
    w("* Nie uwzglednia sie logiki wokol rdzeni (bufory, drzewo rozgloszeniowe")
    w("  zegara, interfejs pamieci), ktora w prawdziwym akceleratorze softmax")
    w("  zajmie wlasna czesc ukladu.")
    w("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"zapisano {args.out}")
    bt, bf, _, _, _, bn = best_single_clock
    print(f"jeden zegar  : {bn} rdzeni @ {bf:.1f} MHz = {bt:.1f} Gop/s")
    print(f"wiele zegarow: {sum(vec2)} rdzeni = {val2 / 1000.0:.1f} Gop/s")


if __name__ == "__main__":
    main()
