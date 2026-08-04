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
| 161.7 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 164.4 |
| 155.8 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 158.4 |
| 117.0 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 119.0 |
| 110.2 | 730 x expe hybrid + 197 x expe cut+retime + 90 x poly4 dsp+retime | 1017 | 112.1 |
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

## 5. Analizy per PIPE_TARGET

Do tej pory kazdy rdzen mial swoja naturalna glebokosc.  W projekcie
docelowym wszystkie sa dopelniane przez `bf16_pipe_pad` do wspolnej
latencji, dzieki czemu staja sie **wymienne 1:1** - to jest dokladnie
teza, ktora sprawdza `make rtl_equivalence_test`.  Ponizej kazdy
PIPE_TARGET jest analizowany jako osobny projekt.

Dwie konsekwencje wyboru targetu:

1. **Ktore rdzenie w ogole wchodza w gre.**  Rdzen o naturalnej glebokosci
   wiekszej niz target nie da sie skrocic, wiec odpada.  Wiersze oznaczone
   w `pipe_target_sweep.csv` jako `za plytki target` sa tu pominiete.
2. **Ile kosztuje dopelnienie.**  Stopnie dopelniajace nie zmieniaja Fmax
   (dFmax = 0 na calym sweepie), ale zjadaja LUT i FF, wiec zmniejszaja
   liczbe kopii, ktore sie mieszcza.

### 5.1. PIPE_TARGET = 4

Rdzenie dostepne przy tej latencji:

| rdzen | stopnie | z tego pad | LUT | FF | DSP | BRAM36 | Fmax [MHz] |
|---|---:|---:|---:|---:|---:|---:|---:|
| expe full-lut | 4 | 0 | 40 | 54 | 0 | 2 | 283.4 |
| expe hybrid | 4 | 0 | 107 | 64 | 0 | 0.5 | 250.3 |

Odpadaja, bo ich naturalna glebokosc przekracza target: `expe cut+retime` (potrzebuje 7 stopni), `poly4+retime` (potrzebuje 9 stopni), `poly4 dsp+retime` (potrzebuje 12 stopni).

**Jeden typ rdzenia:**

| rdzen | zasob wiazacy | N | T [Gop/s] |
|---|---|---:|---:|
| expe full-lut | BRAM36 | 182 | 51.6 |
| expe hybrid | BRAM36 | 730 | 182.7 |

**Mieszanka, wspolny zegar** (rdzenie maja te sama latencje, wiec
mozna je wstawiac zamiennie w jednej domenie zegarowej):

| f [MHz] | sklad | N razem | T [Gop/s] |
|---:|---|---:|---:|
| 283.4 | 182 x expe full-lut | 182 | 51.6 |
| 250.3 | 730 x expe hybrid | 730 | 182.7 |

Optimum: **730 x expe hybrid** przy 250.3 MHz = **182.7 Gop/s** (730 rdzeni).
Optimum jest **dowiedzione**: relaksacja ciagla tego samego zadania
daje to samo ograniczenie, wiec zadna inna kombinacja nie da wiecej.

| zasob | uzyte | dostepne | % |
|---|---:|---:|---:|
| LUT | 78110 | 134600 | 58.0% |
| LUT-as-mem | 0 | 46200 | 0.0% |
| FF | 46720 | 269200 | 17.4% |
| CARRY4 | 0 | 33650 | 0.0% |
| DSP48E1 | 0 | 740 | 0.0% |
| BRAM36 | 365 | 365 | 100.0% |

Wysycone: BRAM36.  Zupelnie nieuzyte: LUT-as-mem, CARRY4, DSP48E1.
Mieszanie **nie oplaca sie** przy tym targecie - najlepszy jest
czysty `expe hybrid`.  Dolozenie wolniejszego typu sciaga wspolny
zegar bardziej, niz zyskuje na liczbie rdzeni.

**Mieszanka, osobne zegary** - gorne ograniczenie.  Uwaga: przy
osobnych zegarach wspolna latencja przestaje cokolwiek znaczyc, wiec
ten wariant przeczy sensowi ustawiania PIPE_TARGET.  Podany dla skali:

730 x expe hybrid @ 250.3 MHz = 730 rdzeni, **182.7 Gop/s**.

### 5.2. PIPE_TARGET = 8

Rdzenie dostepne przy tej latencji:

| rdzen | stopnie | z tego pad | LUT | FF | DSP | BRAM36 | Fmax [MHz] |
|---|---:|---:|---:|---:|---:|---:|---:|
| expe full-lut | 8 | 4 | 40 | 122 | 0 | 2 | 283.4 |
| expe hybrid | 8 | 4 | 107 | 132 | 0 | 0.5 | 250.3 |
| expe cut+retime | 8 | 1 | 241 | 207 | 1 | 0 | 193.2 |

Odpadaja, bo ich naturalna glebokosc przekracza target: `poly4+retime` (potrzebuje 9 stopni), `poly4 dsp+retime` (potrzebuje 12 stopni).

**Jeden typ rdzenia:**

| rdzen | zasob wiazacy | N | T [Gop/s] |
|---|---|---:|---:|
| expe full-lut | BRAM36 | 182 | 51.6 |
| expe hybrid | BRAM36 | 730 | 182.7 |
| expe cut+retime | LUT | 558 | 107.8 |

**Mieszanka, wspolny zegar** (rdzenie maja te sama latencje, wiec
mozna je wstawiac zamiennie w jednej domenie zegarowej):

| f [MHz] | sklad | N razem | T [Gop/s] |
|---:|---|---:|---:|
| 283.4 | 182 x expe full-lut | 182 | 51.6 |
| 250.3 | 730 x expe hybrid | 730 | 182.7 |
| 193.2 | 730 x expe hybrid + 234 x expe cut+retime | 964 | 186.2 |

Optimum: **730 x expe hybrid + 234 x expe cut+retime** przy 193.2 MHz = **186.2 Gop/s** (964 rdzeni).
Optimum jest **dowiedzione**: relaksacja ciagla tego samego zadania
daje to samo ograniczenie, wiec zadna inna kombinacja nie da wiecej.

| zasob | uzyte | dostepne | % |
|---|---:|---:|---:|
| LUT | 134504 | 134600 | 99.9% |
| LUT-as-mem | 3042 | 46200 | 6.6% |
| FF | 144798 | 269200 | 53.8% |
| CARRY4 | 2574 | 33650 | 7.6% |
| DSP48E1 | 234 | 740 | 31.6% |
| BRAM36 | 365 | 365 | 100.0% |

Wysycone: LUT, BRAM36.
Mieszanie oplaca sie: 186.2 Gop/s wobec 182.7 Gop/s dla
samego `expe hybrid`.

**Mieszanka, osobne zegary** - gorne ograniczenie.  Uwaga: przy
osobnych zegarach wspolna latencja przestaje cokolwiek znaczyc, wiec
ten wariant przeczy sensowi ustawiania PIPE_TARGET.  Podany dla skali:

730 x expe hybrid @ 250.3 MHz + 234 x expe cut+retime @ 193.2 MHz = 964 rdzeni, **227.9 Gop/s**.

### 5.3. PIPE_TARGET = 13

Rdzenie dostepne przy tej latencji:

| rdzen | stopnie | z tego pad | LUT | FF | DSP | BRAM36 | Fmax [MHz] |
|---|---:|---:|---:|---:|---:|---:|---:|
| expe full-lut | 13 | 9 | 74 | 112 | 0 | 2 | 283.4 |
| expe hybrid | 13 | 9 | 141 | 122 | 0 | 0.5 | 250.3 |
| poly4 dsp+retime | 13 | 1 | 99 | 177 | 6 | 0 | 194.3 |
| expe cut+retime | 13 | 6 | 273 | 244 | 1 | 0 | 193.2 |
| poly4+retime | 13 | 4 | 179 | 170 | 5 | 0 | 183.8 |

Nic nie odpada - target miesci naturalna glebokosc wszystkich rdzeni.

**Jeden typ rdzenia:**

| rdzen | zasob wiazacy | N | T [Gop/s] |
|---|---|---:|---:|
| expe full-lut | BRAM36 | 182 | 51.6 |
| expe hybrid | BRAM36 | 730 | 182.7 |
| poly4 dsp+retime | DSP48E1 | 123 | 23.9 |
| expe cut+retime | LUT | 493 | 95.2 |
| poly4+retime | DSP48E1 | 148 | 27.2 |

**Mieszanka, wspolny zegar** (rdzenie maja te sama latencje, wiec
mozna je wstawiac zamiennie w jednej domenie zegarowej):

| f [MHz] | sklad | N razem | T [Gop/s] |
|---:|---|---:|---:|
| 283.4 | 182 x expe full-lut | 182 | 51.6 |
| 250.3 | 730 x expe hybrid | 730 | 182.7 |
| 194.3 | 730 x expe hybrid + 123 x poly4 dsp+retime | 853 | 165.7 |
| 193.2 | 730 x expe hybrid + 110 x poly4 dsp+retime + 76 x expe cut+retime | 916 | 177.0 |
| 183.8 | 730 x expe hybrid + 110 x poly4 dsp+retime + 76 x expe cut+retime | 916 | 168.4 |

Optimum: **730 x expe hybrid** przy 250.3 MHz = **182.7 Gop/s** (730 rdzeni).
Optimum jest **dowiedzione**: relaksacja ciagla tego samego zadania
daje to samo ograniczenie, wiec zadna inna kombinacja nie da wiecej.

| zasob | uzyte | dostepne | % |
|---|---:|---:|---:|
| LUT | 102930 | 134600 | 76.5% |
| LUT-as-mem | 12410 | 46200 | 26.9% |
| FF | 89060 | 269200 | 33.1% |
| CARRY4 | 0 | 33650 | 0.0% |
| DSP48E1 | 0 | 740 | 0.0% |
| BRAM36 | 365 | 365 | 100.0% |

Wysycone: BRAM36.  Zupelnie nieuzyte: CARRY4, DSP48E1.
Mieszanie **nie oplaca sie** przy tym targecie - najlepszy jest
czysty `expe hybrid`.  Dolozenie wolniejszego typu sciaga wspolny
zegar bardziej, niz zyskuje na liczbie rdzeni.

**Mieszanka, osobne zegary** - gorne ograniczenie.  Uwaga: przy
osobnych zegarach wspolna latencja przestaje cokolwiek znaczyc, wiec
ten wariant przeczy sensowi ustawiania PIPE_TARGET.  Podany dla skali:

730 x expe hybrid @ 250.3 MHz + 110 x poly4 dsp+retime @ 194.3 MHz + 76 x expe cut+retime @ 193.2 MHz = 916 rdzeni, **218.8 Gop/s**.

### 5.4. Porownanie trzech targetow

| PIPE_TARGET | rdzeni w grze | najlepszy 1 typ [Gop/s] | najlepsza mieszanka [Gop/s] | f [MHz] | N | osobne zegary [Gop/s] |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 182.7 (`expe hybrid`) | **182.7** | 250.3 | 730 | 182.7 |
| 8 | 3 | 182.7 (`expe hybrid`) | **186.2** | 193.2 | 964 | 227.9 |
| 13 | 5 | 182.7 (`expe hybrid`) | **182.7** | 250.3 | 730 | 218.8 |

Sklady optymalnych mieszanek:

* **target 4:** 730 x expe hybrid @ 250.3 MHz -> 182.7 Gop/s
* **target 8:** 730 x expe hybrid + 234 x expe cut+retime @ 193.2 MHz -> 186.2 Gop/s
* **target 13:** 730 x expe hybrid @ 250.3 MHz -> 182.7 Gop/s

**Najlepszy target: 8** - 186.2 Gop/s.

* target 4: 182.7 Gop/s (-1.9% wobec targetu 8)
* target 13: 182.7 Gop/s (-1.9% wobec targetu 8)

Zaleznosc nie jest monotoniczna i warto zrozumiec dlaczego:

* **Za plytki target odcina rdzenie.**  Przy targecie 4 zostaja tylko dwa
  rdzenie i oba sa oparte o BRAM, wiec caly LUT i caly DSP leza odlogiem.
  Przepustowosc jest z gory ograniczona przez 365 blokow BRAM.
* **Za gleboki target tez szkodzi**, ale z innego powodu: dopelnianie
  kosztuje LUT i FF.  Miedzy targetem 8 a 13 `expe hybrid` rosnie ze 107
  do 141 LUT, a `expe cut+retime` z 241 do 273 LUT.  Przy niezmienionym
  Fmax to czysta strata: w tym samym ukladzie miesci sie mniej kopii.
* Optimum lezy tam, gdzie target jest **dokladnie tak gleboki, jak trzeba**,
  zeby wpuscic kolejny rdzen o komplementarnym profilu zasobow.  Tu jest to
  target 8, ktory dopuszcza `expe cut+retime` (7 stopni wlasnych, LUT+DSP)
  obok `expe hybrid` (4 stopnie, BRAM) prawie bez kosztu dopelnienia:
  `cut` potrzebuje tylko 1 stopnia pad, `hybrid` 4.

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

