# Koszt dopelniania potoku (PIPE_TARGET)

Wygenerowane przez `make pipe_target_table`. Nie edytowac recznie.

Uklad `xc7a200tfbg484-1`, Vivado 2025.2, ograniczenie zegara 2.0 ns.
Fmax po samej syntezie, bez implementacji.

**PIPE_TARGET tylko dopelnia w gore.** Rdzen glebszy niz target zostaje
na swojej naturalnej glebokosci - nie da sie go skrocic. Takie wiersze sa
oznaczone `za plytki target` i powtarzaja wynik z natywnej glebokosci.

## PIPE_TARGET = 4

| rdzen | stopnie | pad | LUT | CARRY4 | FF | SRL | DSP | BRAM36 | Fmax [MHz] | poziomy | uwaga |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| expe full-lut | 4 | 0 | 40 | 0 | 54 | 0 | 0 | 2 | 283.4 | 1 |  |
| expe hybrid | 4 | 0 | 107 | 0 | 64 | 0 | 0 | 0.5 | 250.3 | 3 |  |
| expe cut+retime | 7 | 0 | 228 | 11 | 190 | 13 | 1 | 0 | 193.2 | 8 | za plytki target |
| poly4+retime | 9 | 0 | 140 | 0 | 134 | 23 | 5 | 0 | 183.8 | 3 | za plytki target |
| poly4 dsp+retime | 12 | 0 | 80 | 0 | 160 | 19 | 6 | 0 | 194.3 | 0 | za plytki target |

## PIPE_TARGET = 8

| rdzen | stopnie | pad | LUT | CARRY4 | FF | SRL | DSP | BRAM36 | Fmax [MHz] | poziomy | uwaga |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| expe full-lut | 8 | 4 | 40 | 0 | 122 | 0 | 0 | 2 | 283.4 | 1 |  |
| expe hybrid | 8 | 4 | 107 | 0 | 132 | 0 | 0 | 0.5 | 250.3 | 3 |  |
| expe cut+retime | 8 | 1 | 228 | 11 | 207 | 13 | 1 | 0 | 193.2 | 8 |  |
| poly4+retime | 9 | 0 | 140 | 0 | 134 | 23 | 5 | 0 | 183.8 | 3 | za plytki target |
| poly4 dsp+retime | 12 | 0 | 80 | 0 | 160 | 19 | 6 | 0 | 194.3 | 0 | za plytki target |

## PIPE_TARGET = 13

| rdzen | stopnie | pad | LUT | CARRY4 | FF | SRL | DSP | BRAM36 | Fmax [MHz] | poziomy | uwaga |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| expe full-lut | 13 | 9 | 57 | 0 | 112 | 17 | 0 | 2 | 283.4 | 1 |  |
| expe hybrid | 13 | 9 | 124 | 0 | 122 | 17 | 0 | 0.5 | 250.3 | 3 |  |
| expe cut+retime | 13 | 6 | 244 | 11 | 244 | 29 | 1 | 0 | 193.2 | 8 |  |
| poly4+retime | 13 | 4 | 140 | 0 | 170 | 39 | 5 | 0 | 183.8 | 3 |  |
| poly4 dsp+retime | 13 | 1 | 80 | 0 | 177 | 19 | 6 | 0 | 194.3 | 0 |  |

## Przyrost wzgledem naturalnej glebokosci

Liczone tylko dla rdzeni, ktore faktycznie zostaly dopelnione.

| rdzen | target | pad | dLUT | dFF | dSRL | dFmax [MHz] |
|---|---:|---:|---:|---:|---:|---:|
| expe full-lut | 8 | +4 | +0 | +68 | +0 | +0.0 |
| expe hybrid | 8 | +4 | +0 | +68 | +0 | +0.0 |
| expe cut+retime | 8 | +1 | +0 | +17 | +0 | +0.0 |
| expe full-lut | 13 | +9 | +17 | +58 | +17 | +0.0 |
| expe hybrid | 13 | +9 | +17 | +58 | +17 | +0.0 |
| expe cut+retime | 13 | +6 | +16 | +54 | +16 | +0.0 |
| poly4+retime | 13 | +4 | +0 | +36 | +16 | +0.0 |
| poly4 dsp+retime | 13 | +1 | +0 | +17 | +0 | +0.0 |

## Dostepne w ukladzie

| LUT | CARRY4 | FF | LUT-as-mem | DSP48E1 | BRAM36 |
|---:|---:|---:|---:|---:|---:|
| 134600 | 33650 | 269200 | 46200 | 740 | 365 |

