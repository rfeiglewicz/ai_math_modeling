# BF16 exp(x) - zasoby i glebokosc potoku

Wygenerowane przez `make resource_table`. Nie edytowac recznie.

Wszystkie rdzenie zsyntezowane z `PIPE_TARGET=0`, czyli **bez dopelniania**
do wspolnej latencji. Kazdy rdzen ma tu swoja naturalna glebokosc; w
projekcie docelowym dziela wspolny zegar i sa dopelniane do
`UNIFIED_PIPE_DEPTH`, co kosztuje dodatkowe SRL i przerzutniki.

## Metodyka

- uklad: `xc7a200tfbg484-1` (Artix-7 200T)
- Vivado 2025.2, `synth_design` bez implementacji
- ograniczenie zegara **2.0 ns (500 MHz)**, celowo nieosiagalne: synteza
  przestaje optymalizowac po spelnieniu ograniczenia, wiec realistyczny
  target zanizylby Fmax
- Fmax liczone z najgorszej sciezki wewnetrznej: `1000 / (okres - slack)`
- zasoby z `report_utilization` (miejsca w ukladzie), nie z `get_cells`
  (prymitywy) - dwa LUT5 dziela jedno miejsce LUT6
- glebokosc potoku raportowana przez sam RTL przy elaboracji
  (`-GREPORT_DEPTH=1`), wiec nie moze sie rozjechac z kodem

**Fmax jest po samej syntezie, bez placement i routingu.** Na ostatnich
zmierzonych sciezkach krytycznych 56-77% opoznienia to *szacowane*
trasowanie, wiec po implementacji te liczby spadna.

## Zasoby (bez dopelniania potoku)

| rdzen | stopnie | LUT | CARRY4 | FF | SRL | DSP48E1 | BRAM36 | RAMB18 | Fmax [MHz] | poziomy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp2 baseline | 7 | 901 | 25 | 248 | 2 | 4 | 1 | 2 | 48.9 | 17 |
| exp2 opt | 7 | 584 | 23 | 153 | 11 | 5 | 1 | 2 | 50.6 | 15 |
| exp2 opt+retime | 16 | 654 | 33 | 506 | 16 | 5 | 1 | 2 | 161.7 | 6 |
| expe full-lut | 4 | 40 | 0 | 54 | 0 | 0 | 2 | 0 | 283.4 | 1 |
| expe hybrid | 4 | 107 | 0 | 64 | 0 | 0 | 0.5 | 1 | 250.3 | 3 |
| expe cut | 4 | 254 | 11 | 110 | 0 | 1 | 0 | 0 | 117.0 | 12 |
| expe cut+retime | 7 | 228 | 11 | 190 | 13 | 1 | 0 | 0 | 193.2 | 8 |
| expe poly4 | 7 | 142 | 0 | 101 | 19 | 5 | 0 | 0 | 110.2 | 4 |
| poly4+retime | 13 | 143 | 0 | 189 | 23 | 5 | 0 | 0 | 183.8 | 3 |
| poly4 dsp | 8 | 80 | 0 | 105 | 19 | 6 | 0 | 0 | 155.8 | 1 |
| poly4 dsp+retime | 13 | 82 | 0 | 200 | 23 | 6 | 0 | 0 | 163.9 | 1 |
| **dostepne w ukladzie** | - | **134600** | **33650** | **269200** | **46200** | **740** | **365** | **730** | - | - |

Kolumny: LUT = LUT jako logika; SRL = LUT jako rejestr przesuwny
(osobno, bo to dopelnienie potoku, nie logika); CARRY4 dzieli miejsce
ze slice, wiec limit rowna sie liczbie slice; BRAM36 bywa ulamkowy, bo
pojedynczy RAMB18 to pol kafelka.

## Udzial w ukladzie [%]

| rdzen | LUT | FF | DSP48E1 | BRAM36 |
|---|---:|---:|---:|---:|
| exp2 baseline | 0.669 | 0.092 | 0.541 | 0.274 |
| exp2 opt | 0.434 | 0.057 | 0.676 | 0.274 |
| exp2 opt+retime | 0.486 | 0.188 | 0.676 | 0.274 |
| expe full-lut | 0.03 | 0.02 | 0 | 0.548 |
| expe hybrid | 0.079 | 0.024 | 0 | 0.137 |
| expe cut | 0.189 | 0.041 | 0.135 | 0 |
| expe cut+retime | 0.169 | 0.071 | 0.135 | 0 |
| expe poly4 | 0.105 | 0.038 | 0.676 | 0 |
| poly4+retime | 0.106 | 0.07 | 0.676 | 0 |
| poly4 dsp | 0.059 | 0.039 | 0.811 | 0 |
| poly4 dsp+retime | 0.061 | 0.074 | 0.811 | 0 |

Zaden rdzen nie przekracza **1%** zadnego zasobu ukladu. Przy takich
rozmiarach o wyborze nie decyduje zajetosc, tylko Fmax, liczba DSP i to,
czy chcemy wydac blok pamieci - a te trzy rzeczy wykluczaja sie wzajemnie.

## Ile takich rdzeni zmiesci sie w ukladzie

Limit z najciasniejszego zasobu, przy zalozeniu ze rdzenie nie dziela
niczego - w praktyce tablice ROM sa identyczne, wiec dalyby sie
wspoldzielic i realna liczba bylaby wieksza.

| rdzen | limit LUT | limit FF | limit DSP | limit BRAM | **max sztuk** |
|---|---:|---:|---:|---:|---:|
| exp2 baseline | 149 | 1085 | 185 | 365 | **149** |
| exp2 opt | 230 | 1759 | 148 | 365 | **148** |
| exp2 opt+retime | 205 | 532 | 148 | 365 | **148** |
| expe full-lut | 3365 | 4985 | - | 182 | **182** |
| expe hybrid | 1257 | 4206 | - | 730 | **730** |
| expe cut | 529 | 2447 | 740 | - | **529** |
| expe cut+retime | 590 | 1416 | 740 | - | **590** |
| expe poly4 | 947 | 2665 | 148 | - | **148** |
| poly4+retime | 941 | 1424 | 148 | - | **148** |
| poly4 dsp | 1682 | 2563 | 123 | - | **123** |
| poly4 dsp+retime | 1641 | 1346 | 123 | - | **123** |

## Koszt ujednolicenia latencji

Najglebszy rdzen ma **16** stopni, wiec przy wspolnym zegarze kazdy
inny musi zostac dopelniony do tej samej liczby. Dopelnienie idzie w
SRL16E (16 bitow na jeden LUT), wiec jest tanie, ale nie darmowe:

| rdzen | stopnie wlasne | do dopelnienia |
|---|---:|---:|
| exp2 baseline | 7 | 9 |
| exp2 opt | 7 | 9 |
| exp2 opt+retime | 16 | 0 |
| expe full-lut | 4 | 12 |
| expe hybrid | 4 | 12 |
| expe cut | 4 | 12 |
| expe cut+retime | 7 | 9 |
| expe poly4 | 7 | 9 |
| poly4+retime | 13 | 3 |
| poly4 dsp | 8 | 8 |
| poly4 dsp+retime | 13 | 3 |

