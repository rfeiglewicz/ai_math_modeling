# Maksymalna przepustowosc: ile rdzeni expe zmiesci sie w FPGA

Uklad: **xc7a200tfbg484-1**, Vivado 2025.2, wyniki **po syntezie**
(`synth_design`, bez implementacji).  Wszystkie liczby zasobow i Fmax
pochodza z `build/vivado_compare/resources.csv`, generowanego przez
`make compare_cores`.  Zadna wartosc w tym dokumencie nie jest szacowana.

## Wynik

* **Jeden typ rdzenia:** `expe hybrid`, 730 kopii przy 250.3 MHz -> **182.7 Gop/s** (limit: BRAM36).
* **Mieszanka, wspolny zegar 193.2 MHz:** 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime = 1017 rdzeni -> **196.5 Gop/s**.
* **Mieszanka, osobne zegary:** te same 1017 rdzeni, kazdy typ na wlasnym Fmax -> **238.3 Gop/s**.

Mieszanka wygrywa, bo rdzenie wysycaja **rozne** zasoby: `hybrid` konczy
sie na BRAM, `cut` na LUT, `poly4 dsp` na DSP.  Sam `expe hybrid` wysyca
BRAM36 w 100%, ale zostawia wolne 42% LUT, 83% FF, 100% DSP48E1.

Te liczby zakladaja 100% wysycenia ukladu.  Realistyczny wariant (80%)
jest w rozdziale 4.  Powyzsze rdzenie maja **rozna latencje**; wersje
wyrownane do wspolnej latencji (PIPE_TARGET 4 / 8 / 13), czyli takie,
ktore da sie wstawiac zamiennie, sa przeanalizowane osobno w rozdziale 5.

## Model

Kazdy rdzen jest w pelni potokowy: interfejs AXI-Stream przyjmuje nowa
probke w kazdym takcie i po `PIPE_DEPTH` taktach oddaje wynik, bez przerw.
Stad **glebokosc potoku nie wplywa na przepustowosc, tylko na latencje**.
Przepustowosc jednej kopii to dokladnie 1 operacja na takt.

Dla N kopii w jednej domenie zegarowej:

    T = N * f,   f = min Fmax po uzytych typach rdzeni
    N ogranicza najciasniejszy zasob: N = min_r floor(dostepne_r / uzyte_r)

Uwaga o LUT: w raporcie Vivado `LUT as Logic` i `LUT as Memory` (kolumna
`srl`) czerpia z tej samej puli 134600 Slice LUT, dlatego w tabelach nizej
`LUT` oznacza sume `lut_logic + srl`.

## 1. Jeden typ rdzenia, jedna domena zegarowa

| rdzen | LUT | FF | DSP | BRAM36 | Fmax [MHz] | zasob wiazacy | N kopii | T [Gop/s] |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| exp2 baseline | 903 | 248 | 4 | 1 | 48.9 | LUT | 149 | **7.3** |
| exp2 opt | 595 | 153 | 5 | 1 | 50.6 | DSP48E1 | 148 | **7.5** |
| exp2 opt+retime | 670 | 506 | 5 | 1 | 161.7 | DSP48E1 | 148 | **23.9** |
| exp2 optim | 491 | 138 | 2 | 0.5 | 80.4 | LUT | 274 | **22.0** |
| optim+retime | 402 | 200 | 2 | 0.5 | 115.5 | LUT | 334 | **38.6** |
| optim max-retime | 406 | 272 | 2 | 0.5 | 164.8 | LUT | 331 | **54.5** |
| optim half-up | 405 | 272 | 2 | 0.5 | 177.5 | LUT | 332 | **58.9** |
| expe full-lut | 40 | 54 | 0 | 2 | 283.4 | BRAM36 | 182 | **51.6** |
| expe hybrid | 107 | 64 | 0 | 0.5 | 250.3 | BRAM36 | 730 | **182.7** |
| expe cut | 254 | 110 | 1 | 0 | 117.0 | LUT | 529 | **61.9** |
| expe cut+retime | 241 | 190 | 1 | 0 | 193.2 | LUT | 558 | **107.8** |
| expe poly4 | 161 | 101 | 5 | 0 | 110.2 | DSP48E1 | 148 | **16.3** |
| poly4+retime | 163 | 134 | 5 | 0 | 183.8 | DSP48E1 | 148 | **27.2** |
| poly4 dsp | 99 | 105 | 6 | 0 | 155.8 | DSP48E1 | 123 | **19.2** |
| poly4 dsp+retime | 99 | 160 | 6 | 0 | 194.3 | DSP48E1 | 123 | **23.9** |

Najlepszy pojedynczy typ: **expe hybrid** - 730 kopii przy 250.3 MHz = **182.7 Gop/s**, ograniczony przez BRAM36.

### Wydajnosc na jednostke zasobu

To jest liczba, ktora decyduje o skladzie mieszanki: ile operacji na
sekunde daje rdzen w przeliczeniu na jeden zajety element danego typu
(`Fmax / uzyte_r`).  Im wiecej, tym oplacalniej wydac na niego ten zasob.

| rdzen | Mop/s na LUT | Mop/s na DSP | Mop/s na BRAM36 |
|---|---:|---:|---:|
| exp2 baseline | 0.1 | 12.2 | 48.9 |
| exp2 opt | 0.1 | 10.1 | 50.6 |
| exp2 opt+retime | 0.2 | 32.3 | 161.7 |
| exp2 optim | 0.2 | 40.2 | 160.8 |
| optim+retime | 0.3 | 57.8 | 231.0 |
| optim max-retime | 0.4 | 82.4 | 329.6 |
| optim half-up | 0.4 | 88.8 | 355.0 |
| expe full-lut | 7.1 | - | 141.7 |
| expe hybrid | 2.3 | - | 500.6 |
| expe cut | 0.5 | 117.0 | - |
| expe cut+retime | 0.8 | 193.2 | - |
| expe poly4 | 0.7 | 22.0 | - |
| poly4+retime | 1.1 | 36.8 | - |
| poly4 dsp | 1.6 | 26.0 | - |
| poly4 dsp+retime | 2.0 | 32.4 | - |

## 2. Mieszanka rdzeni, jedna domena zegarowa

Rdzenie wysycaja **rozne** zasoby: `hybrid` i `full-lut` blokuja BRAM,
`cut` i `poly4` blokuja DSP, `exp2` blokuje LUT.  Mieszanie typow pozwala
wykorzystac zasoby, ktore jeden typ zostawia niewykorzystane.  Cena jest
taka, ze wspolny zegar spada do Fmax najwolniejszego uzytego typu.

| f [MHz] | sklad | N razem | T [Gop/s] |
|---:|---|---:|---:|
| 283.4 | 182 x expe full-lut | 182 | 51.6 |
| 250.3 | 730 x expe hybrid | 730 | 182.7 |
| 194.3 | 730 x expe hybrid + 123 x poly4 dsp+retime | 853 | 165.7 |
| 193.2 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 196.5 |
| 183.8 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 186.9 |
| 177.5 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 180.5 |
| 164.8 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 167.6 |
| 161.7 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 164.4 |
| 155.8 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 158.4 |
| 117.0 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 119.0 |
| 115.5 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 117.5 |
| 110.2 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 112.1 |
| 80.4 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 81.8 |
| 50.6 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 51.5 |
| 48.9 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 49.7 |

**Optimum przy jednym zegarze: 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime**, razem 1017 rdzeni przy 193.2 MHz = **196.5 Gop/s**.

Ponizej 193.2 MHz tabela nie rosnie: mieszanka juz wysyca BRAM i DSP w
100%, a LUT w 99.9%, wiec zwolnienie zegara nie odblokowuje zadnych
dodatkowych kopii - tylko obniza T.

Wykorzystanie zasobow w tym punkcie:

| zasob | uzyte | dostepne | % |
|---|---:|---:|---:|
| LUT | 134497 | 134600 | 99.9% |
| LUT-as-mem | 4271 | 46200 | 9.2% |
| FF | 98550 | 269200 | 36.6% |
| CARRY4 | 2167 | 33650 | 6.4% |
| DSP48E1 | 737 | 740 | 99.6% |
| BRAM36 | 365 | 365 | 100.0% |

## 3. Mieszanka rdzeni, osobne domeny zegarowe

Jesli kazdy typ dostanie wlasny zegar (rdzenie sa niezalezne, lacza je
tylko strumienie danych), to nie trzeba placic za najwolniejszy typ i
maksymalizuje sie `sum(N_c * Fmax_c)`.

Sklad: **730 x expe hybrid @ 250.3 MHz + 197 x expe cut+retime @ 193.2 MHz + 90 x poly4 dsp+retime @ 194.3 MHz**

Razem 1017 rdzeni, **238.3 Gop/s**.

| zasob | uzyte | dostepne | % |
|---|---:|---:|---:|
| LUT | 134497 | 134600 | 99.9% |
| LUT-as-mem | 4271 | 46200 | 9.2% |
| FF | 98550 | 269200 | 36.6% |
| CARRY4 | 2167 | 33650 | 6.4% |
| DSP48E1 | 737 | 740 | 99.6% |
| BRAM36 | 365 | 365 | 100.0% |

## 4. Realistyczny budzet (80% ukladu)

Punkty z rozdzialow 2 i 3 zakladaja 100% wysycenia ukladu, czego nie da
sie zrutowac.  Ponizej to samo przy budzecie zasobow ograniczonym do
80% kazdego zasobu.

| scenariusz | sklad | N razem | T [Gop/s] |
|---|---|---:|---:|
| jeden zegar 193.2 MHz | 584 x expe hybrid + 157 x expe cut+retime + 72 x poly4 dsp+retime | 813 | 157.1 |
| osobne zegary | 584 x expe hybrid + 157 x expe cut+retime + 72 x poly4 dsp+retime | 813 | 190.5 |

## Zastrzezenia

* Fmax jest **po syntezie, bez place & route**.  Na ostatnich sciezkach
  krytycznych 56-77% opoznienia to *szacowane* trasowanie.  Przy wysyceniu
  ukladu bliskim 100% realne Fmax bedzie wyraznie nizsze - stad rozdzial 4.
* Liczby kopii z rozdzialow 2, 3 i 5 to **gorne ograniczenie**, a nie
  projekt, ktory da sie zbudowac.  "Optimum dowiedzione" w rozdziale 5
  znaczy tylko, ze przy tych zmierzonych kosztach zasobow nie ma lepszego
  skladu - nie, ze taki uklad sie zrutuje.
* Zawartosc ROM jest identyczna we wszystkich kopiach.  BRAM w Artix-7 jest
  prawdziwie dwuportowy, wiec jeden blok moze obsluzyc 2 rdzenie na takt;
  to podnioslo by limit dla `hybrid` i `full-lut` o czynnik 2, kosztem
  dodatkowej logiki adresowej.  Nie jest to zaimplementowane w RTL.
* Nie uwzglednia sie logiki wokol rdzeni (bufory, drzewo rozgloszeniowe
  zegara, interfejs pamieci), ktora w prawdziwym akceleratorze softmax
  zajmie wlasna czesc ukladu.

