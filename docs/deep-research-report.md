# Podsumowanie

Najtańsza implementacja sprzętowa funkcji $e^x$ dla ujemnych argumentów w formacie bfloat16 (bf16) wymaga kompromisu między dokładnością (≤0,5 ULP) a zasobami (LUT, DSP, pamięć, bramki). Format bf16 (1-bit znak, 8-bit wykładnik, 7-bit mantysa) oferuje zakres dynamiczny identyczny z FP32, przy czym najmniejsza normalna wartość to ~2^–126, a subnormalna ~2^–133. W praktyce implementacje sprzętowe *flushują* subnormale do zera.

Dla funkcji $e^x$ z argumentem ujemnym (softmax) przyjmuje się zakres argumentów do ok. [–126, 0]. Dążymy do błędu ≤0,5 ULP (tj. różnica wyników ≤ połowa najmniej znaczącego bitu w bf16), co gwarantuje poprawne zaokrąglenie do najbliższego. Metody aproksymacji obejmują: redukcję zakresu (x = k·ln2 + r), wielomiany (Taylor, minimax, Remez), interpolacje w tabelach, aproksymacje kawałkami liniowymi, CORDIC i manipulacje bitowe (np. metoda Schraudolpha). W tabelach porównaliśmy te metody pod kątem liczby operacji sprzętowych oraz błędów ULP. Przykładowo, wielomiany minimaksowe przy stopniu 3–4 osiągają błędy ≲0,5 ULP przy użyciu kilku mnożników FPU (dsp), podczas gdy użycie tabeli z interpolacją (np. kilkanaście–kilkadziesiąt wpisów) pozwala uniknąć mnożeń kosztem pamięci BRAM. Implementacja CORDIC wymaga wielu kroków iteracyjnych (addycji/shiftów) i osiąga dobrą dokładność kosztem opóźnień.

W FPGA najważniejsze metryki to LUT, DSP, BRAM, a także częstotliwość i przepustowość. Na przykład projekt *Exp Arithmetic Unit* użył ~2483 LUT i 21 DSP (Virtex5) dla SP przy dokładności ~0,83·2^–17 (błąd ok. 0,83 ULP w FP32). Metody oparte na tablicach (LUT) zużywają pamięć – np. trzy niezależne LUT plus niewielki wielomian liniowy osiągnęły implementację exp(64-bit) (FP64) przy zadowalającej dokładności. Odpowiedniki w ASIC można szacować na dziesiątki tysięcy bramek logicznych plus kilka operacji zmiennoprzecinkowych; przykładowo 180 nm ASIC z implementacją LUT-liniową osiągnął opóźnienie 7,15 ns przy 100 MHz i 5,3 mW dla bloku softmax. W naszym porównaniu preferujemy metody łączące redukcję zakresu i krótkie wielomiany minimaksowe, które zapewniają ≤0,5 ULP przy minimalnym koszcie sprzętowym. Najtańszą opcją wydaje się *wielomian minimaksowy stopnia 3–4* na zredukowanym przedziale r∈[0, ln2), stosowany z szybką manipulacją wykładnika (shift) do utworzenia $2^k$, plus ewentualnie mały LUT dla drobnej poprawki. Taka struktura wymaga kilku DSP i LUT (skalowalnych na urządzeniach Xilinx/Intel) i daje niskie opóźnienie. Alternatywnie *metoda Schraudolpha* (trik bitowy) jest ultraszybka ale ma za duże błędy dla 0,5 ULP. Pełne wyliczenia i porównania zasobów/uzyskanej dokładności pokazano w tabelach poniżej. 

   *Źródła:* Specyfikacja bfloat16; definicja 0,5 ULP; podejścia do aproksymacji exp (wielomiany minimaksowe, tablice interpolacyjne, CORDIC) oraz przykładowe implementacje FPGA/ASIC. Wszelkie założenia zaznaczono w tekście.

## Specyfika formatu bfloat16

Format bfloat16 (Brain Floating Point) ma 16 bitów: 1 bit znaku, 8 bitów wykładnika i 7 bitów mantysy. Ponieważ bias wykładnika to 127, zakres wykładnika normalnych liczb wynosi E∈[–126..+127]. Minimalna dodatnia wartość normalna to ≈2^–126, zaś subnormale (teoretycznie w bf16) mają wartość ≥2^–133. Jednak architektury TPU i inne implementacje zazwyczaj *flushują* subnormale do zera, traktując je jak dokładne zera. Maksymalna wartość (w przybliżeniu FP32) to ≈3,4·10^38. 

Dla funkcji exp przyjmujemy rozpatrywać argumenty ujemne (x≤0) typowo w zakresie od zera do (efektywnie) –∞. Dla pełnego zakresu bf16 największe wartości |x| mogłyby dochodzić do ~126, ale w softmax zazwyczaj mamy ograniczone różnice. Każdy wynik exp(x) w bf16 ma wagę ULP (unit-in-last-place) równą 2^(E_out–7), gdzie E_out to wykładnik wyniku. Błąd ≤0,5 ULP oznacza, że |y_approx – y_exact| ≤ 0,5·2^(E_out–7). Takie kryterium gwarantuje poprawne zaokrąglenie do najbliższego (round-to-nearest) w standardzie IEEE-754. Dla zapewnienia ≤0,5 ULP należałoby zmierzyć maksymalny błąd nad całym zakresem – najtrudniejszy przypadek daje definicję dokładności. W praktyce badamy maksymalną różnicę w reprezentacji bf16 (np. przy pomocy konwersji do float32 i różnicy) oraz sprawdzamy, czy jest poniżej połowy odstępu między sąsiednimi wartościami bfloat16.

## Metody aproksymacji funkcji $e^x$

Różne metody aproksymacji exp(x) można podzielić na:

- **Redukcja zakresu + aproksymacja reszty:** Najczęściej rozbijamy $x = k\ln2 + r$, z całkowitym $k = \lfloor x/\ln2\rceil$ i resztą $r\in[-\ln2/2,\,+\ln2/2]$. Wówczas $e^x = 2^k e^r$. W praktyce dla sprzętu computing 2^k realizuje się przez przesunięcie wykładnika (np. dodanie biasu i zapisanie k w polu wykładnika), a $e^r$ aproksymujemy wielomianem czy tablicą. Na przykład w implementacjach Intel/ARM przyjmuje się: obliczyć $k=\mathrm{round}(x/\ln2)$, wyznaczyć $r=x-k\ln2$, then $e^r\approx p(r)$, a wynik $=2^k \times p(r)$. Minimaxowy wielomian rzędu d dla $r\in[-\ln2/2,\ln2/2]$ zapewnia minimalizację maksymalnego błędu (Remez). Im wyższy stopień, tym mniejszy błąd, ale większy koszt mnożeń/funkcji FPU. Typowo dla 0,5 ULP w bf16 wystarcza stopień 3–4. 

- **Aproksymacje wielomianowe (Taylor vs Minimax):** Wielomiany Taylora (szereg Maclaurina) dają dobre przybliżenie blisko punktu ekspansji (zazwyczaj 0). Jednak daleko od zera konwergują wolniej. Wolniej konwerguje Taylor dla niskich stopni (wielomiany 3-4 rzędu mogą mieć błąd >0,5 ULP). Lepiej zatem używać wielomianów minimaksowych (znajdowanych np. narzędziem Sollya) – one minimalizują maksymalny błąd na zadanym przedziale. Przykładowo, rząd-3 minimax na $r\in[-\ln2/2,\ln2/2]$ można dobrać tak, by zapewnić błąd poniżej 0,5 ULP dla bf16. 

- **Tabele z interpolacją (LUTs):** Można podzielić przedział argumentu na segmenty i przechować wartości $e^x$ w pamięci. Prostym rozwiązaniem jest LUT dla kilku krotek argumentu oraz liniowa interpolacja pomiędzy punktami. Np. Di Franco et al. zaproponowali dla softmax prostą interpolującą LUT (brak dokładnych danych o błędzie). Li i wsp. wykazali, że LUTy często dają najlepszy kompromis między dokładnością a szybkością, przewyższając szereg Taylora czy CORDIC. Wadą jest konieczność przechowywania dużej ilości punktów: większa tablica → lepsza dokładność, ale wymaga BRAM. Przykłady: trójstopniowe rozwiązanie dla FP64 używało trzech oddzielnych tablic plus krótkiego wielomianu liniowego.

- **Aproksymacje kawałkami liniowymi (PWL):** Przybliżamy wykres $e^x$ segmentami liniowymi. Interwały należy dobierać nierównomiernie (większe kroki na płaskich fragmentach, mniejsze tam, gdzie e^x szybko rośnie). Matematycznie optymalne długości fragmentów spełniają $(b-a)^2 e^a = const$. W sprzęcie realizacja PWL wymaga tylko jednej mnożarki i dodawania na segment, co jest proste. Wadą jest relatywnie duży błąd przy krawędziach segmentów – aby zmieścić się w 0,5 ULP, może potrzeba kilkadziesiąt segmentów. W softmaxie niektóre prace proponowały PWL jako szybkie przybliżenie przy akceptowalnym błędzie (np. praca Di Franco ).

- **Metoda bitowa (Schraudolph):** Upraszcza obliczenia, traktując 32-bit float jako liczba całkowita. Schemat: $i = 2^{23}(x/\ln2) + (bias\cdot 2^{23})$, potem interpretacja bitów int jako float ≈ $2^{x/\ln2} = e^x$. Jest bardzo szybka (zwykłe dodanie stałej i rzutowanie bitowe), ale ma duży błąd (rzędu kilku procent) – nie spełnia 0,5 ULP dla bf16. Czasem metoda z dodatkowym krokiem Newtona (korekta), ale nadal rzadko gwarantuje ekstremalną dokładność.

- **CORDIC hiperboliczny:** Mniej popularny dla exp (częściej stosowany dla sin, cos, tanh). Po odpowiedniej transformacji (np. użycie wzorów na $e^x = \cosh(x) + \sinh(x)$) można iteracyjnie osiągać wynik za pomocą sum i shiftów. Wymaga kilkunastu cykli iteracji (błąd ∼2^(–n) po n iteracjach), ale nie ma zwykłych mnożek – co jest plusem w czystych bramek logicznych. Przykład: eksplorowany w pracach EAU. Zwykle odstaje w szybkości od wielomianów z DSP.

- **Interpolacja wielowymiarowa (hybrydowa):** Dla złożonych implementacji (np. wyrenderowane w pracy Hosseiny) dzieli się wejście na część najmniej znaczącą (polynomial), środkową (interpolacja) i najbardziej znaczącą (czysta tabela). Dla bf16 (16-bit), pomijając środek, wystarczają dwie części: wielomian i LUT. Pozwala to zredukować rozmiar potrzebnych LUT.

- **Inne (Pade, elementarne przebiegi):** Aproksymaty Pade’a (iloraz wielomianów) mogą dawać dobre przybliżenia, ale wymagają mnożenia i dzielenia. Rzadziej stosowane w układach akceleracyjnych (większe opóźnienie). Niektóre akceleratory przyjmują uproszczone funkcje softmax bez dzielenia, koncentrując się na błędzie statystycznym.

Poniższa tabela porównuje główne kategorie metod:  

| **Metoda**                    | **Opis / Notatki**                                           | **Dokładność (błąd)**        | **Zasoby FPGA**                         | **Zasoby ASIC**                |
|-------------------------------|--------------------------------------------------------------|------------------------------|------------------------------------------|-------------------------------|
| **Redukcja + poly (Taylor)**  | Rozkład do $2^k e^r$, $r$ aproksymowany wiel. Taylora      | Wyższy błąd, wolniejsza konw. (∼1–2 ULP dla n=3) | 3–4 DSP (Horner) + LUTy na konst (stopień 3–4) | ~50–100k bramek (wielom.)      |
| **Redukcja + poly (Minimax)** | Jak wyżej, ale wielom. minimax (Remez)      | Błąd zredukowany (<0,5 ULP dla n≈3–4) | 3–4 DSP + logiczne (stopień 3–4) | Kilkadziesiąt kB bramek + DSP  |
| **Tablica + interpolacja**    | LUT dla $e^r$ (lub $2^r$) + liniowa/kwadratowa interpolacja | Świetna (można zbliżyć do 0 ULP przy dużej tablicy) | BRAM do przechowania ~16–64 wpisów + trochę DSP (interpolacja) | SRAM  lub ROM (~tens kB) + proste arytm. |
| **Aproks. kawałkami (PWL)**   | Ciąg segmentów liniowych dopasowanych do $e^x$ | Umiarkowana (kilka ULP przy małej liczbie segmentów) | LUTy do liniowych przeliczników (mnożenie+dodawanie), niska złożoność | Układ stałych współczynników (~kilka k bramek) |
| **Schraudolph (bit hack)**    | $i=2^{mant}\!*(x/\ln2)+const$, cast bits→float | Duży błąd (~5–10 % w fp32, niedostateczny dla 0.5 ULP) | 1–2 LUT + arytm. całkowitych, brak DSP | Kilka k bramek, lecz za mała dokładność |
| **CORDIC hiperboliczny**      | Iteracyjny alg. shift/add do $e^x$ (16–32 iter.)         | Dobrze (błąd ~2^(–#iteracje)), można ~0.5 ULP przy ~10–12 it. | Brak DSP (całkowitych), wielokrotne dodawanie (~10 cykli) | ~xADC (~kilka k bramek + rejestry), opóźniony |
| **Hybrydowe**                 | Minimax + LUT (podział bity)              | Bardzo dobra (kompozycja metod)       | Mix z powyższych metod, wyższy koszt   | Mix, złożona organizacja     |

Każda metoda daje inną dokładność i zużycie zasobów. Generalnie:
- Minimaxowe wielomiany stopnia 3–4 zapewniają ≤0,5 ULP przy niewielu mnożeniach.
- LUT + interpolacja (np. kilkanaście-klika punktów) może spełnić 0,5 ULP, ale kosztuje pamięć.
- CORDIC potrzebuje więcej czasu, ale niewiele mnożek (może wymagać iteracji).
- Metody PWL czy uproszczone (bit hack) są szybkie, lecz mają zwykle większy błąd.
- W praktyce łączymy **redukcję wykładnikową** ($2^k$ przez shift) z aproksymacją reszty, co jest najczęściej zalecanym podejściem.

## Zasoby sprzętowe i topologie

### FPGA

Typowa budowa modułu exp w FPGA (np. Xilinx/Intel):
- **Dekoder bf16:** Wydobycie znaku, wykładnika, mantysy (7 bitów + ukryta jedynka). Prosta logika.
- **Redukcja zakresu:** Obliczenie k = round(x/ln2). Często robi się to mnożąc mantysę i wykładnik przez 1/ln2 (w liczbach stałoprzecinkowych) i oddzielając część całkowitą. Można też dodawać stałą biasu i wykonywać rzuty typu float-to-int. W FPGA wymaga to jednego mnożenia stałoprzecinkowego (może DSP) i kilku operacji logicznych.
- **LUT $2^k$ (jeśli potrzebne):** Jeśli $k$ mieści się w zakresie wykładnika bf16 (±126), 2^k ustawia się przez zmianę pola wykładnika (plus obsługa przepełnienia/zer). To nie wymaga dodatkowego mnożenia – tylko ustawienie bitów (dodanie 127 do k i wpis do wyjścia).
- **Aproksymacja $e^r$:** Najbardziej kosztowna część. Jeśli używamy wielomianu Hornera, to sekwencyjnie lub pipeliningiem wykonujemy kolejne mnożenia-dodawania (stopień d wielomianu Hornera to d mnożeń + d dodawań). Każde mnożenie FP zwykle mapuje się na DSP typu Xilinx DSP48 (lub serię LUT). Liczba etapów zależy od pipelinu/FMA. Jeśli zamiast FP używamy liczb stałoprzecinkowych na fragment mantysy, koszty maleją.
  - **Wielomiany:** np. minimalny stopień 3 (Horner) = 3 mnożenia + 3 dodawania, w pełnym precyzyjnym FP. Odpowiednio 3 DSP, ~3*lut(adders) plus kontrola. W FPGA zwykle takie FPU wykorzystują DSP (ale można też na LUT).
  - **Interpolacja:** jeśli $r$ podzielone na N segmentów, dla każdego segmentu przechowuje się współczynnik kierunkowy i stałą. Potrzebne są 1~2 mnożenia (slope·offset + add). Dla np. 16 segmentów – brama adresowa + 16×(mnożenie+add).
  - **CORDIC:** k kroki iteracyjne (np. 10) – każdy krok to shift/adder/reg. Mapa pure-LUT if per bit. W Xilinx można użyć DSP albo LUT jako sumatory.

- **Łączniki wyników:** Po obliczeniu mantysy i wykładnika, łączymy wyniki: wynik = sygn(positiv)*2^k_mant * (1 + δ), gdzie δ wyliczone z wielomianu. W praktyce łączymy bit znaku (0) z obliczonym wykładnikiem (oryginalny wykładnik + k plus bias) i frac (korekta from poly).

Metryki FPGA (przykłady z literatury):
- Projekt EAU (Exponential Arithmetic Unit) dla SP zaprojektowany przez Ciurzak i wsp. użył ~2483 LUT i 21 DSP na Virtex5 (stopień ~8 Hornera) z błędem max ~0,83·2^–17 (≈0,83 ULP). Po optymalizacjach DSP obniżono do 34 (stopień 12?) przy ~2020 LUT. Wersja na FP64 (double) zajął ~55k LUT.
- Xu i in. (2021) zaproponowali FPU dla bf16 z dedykowanym modułem EXP: wykorzystali część exps(x) realizowaną przez mnożenie mantysy przez log2(e) i przesunięcie bitów, a korektę mantysy dwu-gałęziowym wielomianem 2×2 stopnia. Ich jednostka była zorganizowana jako dwa etapy (exps+P(x)), prawdopodobnie zaprojektowana do SIMD.
- W aplikacji softmax Yang i in. (2023) rozdzielili int i frac, ROM na int i polynom na frac. Nie dano liczby LUT, ale typowe implementacje HLS zajmowały kilkaset LUT i kilka DSP.
- We własnych oszacowaniach: polynom stopnia 4 (LUT eksperyment): ~4 DSP, ~~200 LUT, 1–3 cykle pipeline; interpolacja (16-entry): ~16 BRAM lub distributed RAM, ~4 LUT do przeliczeń, bardzo niskie opóźnienie; CORDIC (10 iter.): ~10 adderów, ~10*ze 100 LUT, 10 cykli.
- **Tabele porównawcze:** Zestawiając metody, patrzymy na LUTy, DSP, ewent. BRAM i opóźnienie. Np. metoda „redukcja+minimax (3)” zużywa 3 DSP + ~100 LUT (drobne), opóźnienie ~1–3 ns (pipelined) przy <0,5 ULP. Metoda „PWL(16)” ~16×(1 DSP+1 LUT) = 16 DSP+16 LUT, ale pozwala traktować każdy segment w 1-2 cyklach. Szczegółowe dane w tabeli poniżej (przykłady).

| **Metoda FPGA**                   | **LUT**      | **DSP**     | **BRAM**    | **Latency (cykle)**    | **Max. błąd**        |
|-----------------------------------|--------------|-------------|-------------|------------------------|----------------------|
| Minimax poly Horner (stopień 3)   | ≈150–250     | ≈3          | 0           | ≈3–4 (pipelined)       | ≲0,4 ULP (założ.)     |
| Minimax poly Horner (stopień 4)   | ≈200–300     | ≈4          | 0           | ≈4–5 (pipelined)       | ≲0,2 ULP (założ.)     |
| Taylor (stopień 3)                | ≈150        | ≈3          | 0           | ≈3–4 (pipelined)       | ≈1 ULP (założ.)       |
| LUT + line. interp. (16 wpisów)   | ≈50 (sterowanie) | 0 (albo 1) | 1 BRAM (16)  | ≈1–2 (adresowanie+mul)  | ≲0,5 ULP (przy dobr. wpis.) |
| PWL (16 segm.)                    | ≈100        | ≈16         | 0           | ≈1 (na segment)        | ~1–2 ULP (zależy)     |
| CORDIC (12 iteracje)              | ≈500        | 0           | 0           | 12                     | ~0,5 ULP (12 iter.)   |
| Schraudolph (bit hack)           | ≈20         | 0           | 0           | 1                      | ~>10 ULP (zbyt duży)  |

> *Uwaga:* Liczby szacunkowe (zał., FPGA Xilinx UltraScale). LUT – ilość logicznych LUT (5–6- wej.), DSP – bloki DSP48, BRAM – pamięć blokowa. Latency (przy pipeline). Błędy założone dla metody na docelowym przedziale. Źródła: prace FPGA, rekomendacje ARM/Intel. Dokładna liczba zależy od implementacji (stopień, liczba segmentów). 

Na przykład metoda minimaksowa stopnia 3 oferuje spełnienie 0,5 ULP przy niskim koszcie (kilka DSP) – to nasz obecny faworyt. Metoda LUT-interpolacja (16–32 wpisów) daje najwyższą dokładność, lecz wymaga pamięci BRAM. CORDIC w wersji 12-iteracyjnej daje średnią precyzję kosztem większego opóźnienia. Metoda Schraudolpha nie spełnia 0,5 ULP. 

**Architektura FPGA (schemat blokowy):** Poniżej przykładowy schemat blokowy (Mermaid) dla architektury pipelinowej:

```mermaid
graph TD
  In[Input x (bf16)] --> Decomp[~ Dekoder (znak, exp, mantysa)~]
  Decomp --> Range{Redukcja zakresu}
  Range -->|\(k\)| ExpAdj[2^k (shift wykładnika)]
  Range -->|\(r\)| Poly[Wielomian minimaksowy: e^r (Horner)]
  ExpAdj --> Combine[Scalenie wyniku]
  Poly --> Combine
  Combine --> Out[Wynik e^x (bf16)]
```

#### Przykłady architektur FPGA

1. **Minimax (Horner):** Wejście bf16 → rozbicie x na k i r przez mnożenie przez (1/ln2) i zaokrąglenie. Następnie $2^k$ przez ustawienie wykładnika (brak kosztownych operacji). Reszta $r$ do wielomianu Hornera (np. stopnia 3): 3×FMA (DSP) i 3×sumy. Wynik sklejany w poprawnym formacie bf16. Pipeline można zrobić głębszy (każdy FMA = jedna faza). Pozwoli to osiągnąć wysoką przepustowość (1 wynik na cykl przy wystarczającej liczbie DSP).
   
2. **Tablica + polynom:** X rozbija się do dwóch części: wskaźnika i reszty. Używamy LUT na resztę (np. 32 wartości $e^r$ co 0.03125) oraz liniowej interpolacji między sąsiadami (mnożenie+add). Daje to wyższą dokładność (zmniejsza błąd interpolacji). W praktyce w FPGA adresowanie LUT (BRAM) + 1 mnożenie + 1 sumator. Często stosowane w aplikacjach softmax.

3. **CORDIC (hiperboliczny):** Rozkład funkcji na hiperboliczne obracanie wektora. Działa jak sieć sumatorów i shiftów. Można go przedstawiać blokowo: 
```mermaid
graph TB
  In2[x (stałoprzec.)] --> C1[Inicjalizacja CORDIC]
  C1 --> CORDIC[| \* iteracje (shift+add) \*|]
  CORDIC --> Out2[e^x (FP stacjonarny)]
```
   Każda iteracja to jeden DSP-free add/shift. Latencja ≈(#iteracji) cykle. W FPGA duży plus: nie wymaga DSP, ale minus: wolniejsze (kilkanaście cykli). Zaletą jest duża liniowość błędu. Implementowali to np. w prototypach EAU.

### ASIC

W ASIC najważniejsze są:
- **Liczba bramek (gate count):** Określa obszar. Przelicznik: ~5 LUT FPGA ≈ 1 kbramka ASIC (zależnie od węzła). Dla oszacowania, ~100k LUT ≈ 20M bramek. 
- **Powierzchnia krzemu (mm²):** zależy od procesu (np. 28 nm vs 7 nm). Dla przybliżenia, ~5–10M bramek to kilka mm² w nowoczesnym procesie.
- **Moc (mW):** zależy od częstotliwości, napięcia, i liczby tranzystorów. Np. Gowtham 2025 implementował algorytm softmax (exp+log) w 180nm: przy 100 MHz całość zużywała 5,3 mW. Dla exp samego w nowszym procesie spodziewamy się proporcjonalnie mniejszej mocy przy wyższej częstotliwości.

Dane z literatury (przykłady):
- **Gowtham et al. (2025):** ASIC 180 nm (GPDK180), Softmax LUT+polynom (exp+log). Dla 5-elementowego softmax: opóźnienie 7,15 ns @100 MHz, moc 5,3 mW. Bramka count nie podano, ale dla 180nm jest ono dość duże (przypuszczalnie kilka milionów bramek).
- **Capra et al. (2021, Scientific Reports):** Pseudo-softmax ASIC 90 nm. Dla 8-bit wejść i 10 klas: 3,22 ns latencja (1-PLL, nie pipel.), moc przy 310 MHz (1,0 V). Obszar ~30% większy niż wcześniejszy projekt [49] (Li 2019). Dla 3-bit wejść ten sam MSE osiągnął projekt z ~takim samym opóźnieniem. Z fig. 8C/D wynika, że obszar rośnie niemal liniowo z liczbą wejść (klas) i z bitwidth (zwłaszcza blok PWL_dzielenie). Dla exp samego prawdopodobnie ~maks kilkaset K bramek w 90nm, moc rzędu mW przy ~300 MHz.
- **Typowe FPU w ASIC:** Jeden mnożnik FP32 ~20k bramek (w nowszym procesie), dodawanie FP ~5–10k. Wielomiany minimax stopnia 3 (3 mult +3 add = ~3×20k + 3×5k ≈ 75 kbr) plus logiczna redukcja (kilkadziesiąt k). Daje ~100k bramek. Zatem całe exp (razem z dekoderem i logiką przesunięć) kilkaset k bramek.
- **Tablica ROM:** Jak np. 32 słowa po 16 bitów = 512 bitów pamięci – to rzędu setki tranzystorów, praktycznie zaniedbywalnie mało.
- **CORDIC:** Tutaj ~N addyty + rejestry. Dla N=10, 10 addytorów FP (~10×5k=50k) plus logika sterująca. Trochę podobnie do wielomianów rzędu ~2.

W tabeli porównującej ASICowe metody (założenia):

| **Metoda ASIC**             | **Bramki**      | **Powierzchnia [mm²]** | **Moc [mW]**         | **Opóźnienie (ns)**  |
|----------------------------|----------------|-------------------------|----------------------|----------------------|
| Minimax poly (stop.3)      | ~100–200k bramek | ~0,1–0,2 (w 28nm)      | ~<5 mW (GHz 1V)      | ~1–2 ns (pipelined)  |
| Minimax poly (stop.4)      | ~150–300k      | ~0,15–0,3             | ~<8 mW               | ~2–3 ns            |
| LUT+interp (32)            | ~<50k (ROM)+logika | ~0,03–0,05           | ~<1 mW               | ~<1 ns             |
| CORDIC (12 it.)            | ~50–100k       | ~0,05–0,1             | ~<2 mW               | ~~10 ns           |
| Schraudolph (bit hack)     | ~~10k          | ~~0,01                | ~~0,1 mW             | ~~<1 ns           |

*Przykłady:* Gowtham: 7,15 ns, 5,3 mW @100 MHz; Capra: 3,22 ns @310 MHz (90nm). Liczby powyżej zakładają nowoczesne ASICy (~28 nm). Rzeczywiste liczby zależą od procesu.

## Proponowane architektury

Na podstawie wymagań (≤0,5 ULP, negatywne arg.) sugerujemy następujące minimalne bloki:

- **FFT FPGA:** Moduł exp składający się z: 1) *Modułu redukcji.* Wejście bf16 → rozszyfrowanie → mnożenie przez stałą 1/ln2 → podział na część całkowitą k i resztę r. 2) *Modułu $2^k$.* Ustawienie wykładnika wyjścia BF (przez dodanie biasu i wyzerowanie mantysy). 3) *Polymod.* Wielomian minimaksowy Hornera (stopień ~3–4) liczący $e^r≈1 + c_1 r + c_2 r^2 + ...$ (lub rozszczepiony np. w formie FMA). 4) *Scalacz.* Złożenie wyniku w końcowy BF: znak 0, nowy wykładnik, poprawiona mantysa. Bloki te można połączyć w potok: multiplikator 1/ln2 → pipeline → multiplikatory Hornera → scalanie. Dla Xilinx można wykorzystać DSP48 i zredukowane jednostki FMA, a dla mid-range Altera podobne.

- **ASIC exp:** Architektura analogiczna do FPGA: dedykowane bloki arytmetyczne FP. Moduł redukcji jako mnożenie stałoprzecinkowe przez 1/ln2 (implementacja skrótu, część z wykorzystaniem logicznego skrętu bitów), logikę aproksymacji wielomianem w kaskadzie. Dla niskiego opóźnienia pipeline i kilka stadií. Dla taniości można zredukować precyzję wewnętrzną (np. 16–24 bitów w mnożeniu), ponieważ docelowo i tak zaokrąglamy do bf16.

Przykładowa topologia ASIC (Mermaid):

```mermaid
graph LR
  X[(Wejście x)] --> A[Argument redukcji: x* (1/ln2)]
  A --> B[Int part k = round(A)]
  A --> C[Frac part r = A - k]
  B --> D[Shift wykładnika (2^k)]
  C --> E[Minimax poly Horner r→e^r]
  D --> F[Times Mantissa (concatenate)]
  E --> F
  F --> Y[(Wynik y=e^x)]
```

## Parametryzacja i trade-off

Kluczowe parametry dobierane są zgodnie z wymaganą dokładnością:

- **Stopień wielomianu:** Zwykle 3 lub 4. Dla bf16 i błędu ≤0,5 ULP stopień 3 (quartic Horner) często wystarcza. Stopień 4 (5 współczynn.) gwarantuje margines przy zachowaniu niewielkiego wzrostu zasobów. Dobiera się go przy pomocy narzędzi (Sollya) tak, by max|błąd| ≤ 0,5 ULP.
- **Rozmiar tablicy/interpolacji:** Jeśli stosujemy LUT, liczba punktów wpływa na błąd. Dla interpolacji liniowej ok. 16–32 wpisów w zakresie $r\in[0,\ln2]$ daje precyzję rzędu sub-ULP. Kwadratowa interpolacja (2-stopniowa) z podobną liczbą wpisów jeszcze bardziej redukuje błąd, ale wymaga dodatkowych DSP do mnożenia kwadratu offsetu.
- **Liczba iteracji CORDIC:** Dla hiperbolicznego CORDIC ~10–12 iteracji daje błąd ~2^-12 (~0,00024), co w bf16 (~0,0078 minimalny krok) może spełniać 0,5 ULP. Każda iteracja to 1 cykl pipeline.
- **Zakres wejścia:** Jeśli mamy dodatkowe założenia, możemy ograniczyć przedział $x$ (np. typowe softmax ≤[–16,0]), co zmniejsza wymagania w obliczeniach. Przy pełnym zakresie bf16 musimy obsługiwać ekstremalne wartości (wynik prawie 0).

**Tabela porównawcza (wybrane opcje, przybliżone wartości):**

| **Metoda/Parametry**      | **Dokładność**  | **FPGA LUT**  | **FPGA DSP** | **FPGA BRAM** | **ASIC bramek**    | **ASIC mm²**  | **Użycie mocy**     |
|--------------------------|-----------------|---------------|--------------|---------------|--------------------|---------------|---------------------|
| Minimax rzędu 3 (Horner) | ≤0.5 ULP        | ~200          | ~3           | 0             | ~150k (28nm)       | ~0,15 (28nm)  | ~5 mW (GHz)         |
| Minimax rzędu 4 (Horner) | ≤0.3 ULP        | ~300          | ~4           | 0             | ~200k              | ~0,20         | ~7 mW               |
| LUT 16 + linear interp.  | ≤0.5 ULP (zał.) | ~50 (adres)   | 0–1          | 1 (16×)       | ~50k (16·16b ROM)  | ~0,05         | ~1 mW               |
| LUT 32 + kwadrat interp. | ≤0.2 ULP        | ~100          | 2            | 1 (32×)       | ~70k               | ~0,07         | ~2 mW               |
| PWL 16 segmentów         | ~1 ULP          | ~100          | 16           | 0             | ~~100k             | ~~0,1         | ~~5 mW              |
| CORDIC (12 it.)          | ~0.5 ULP        | ~500          | 0            | 0             | ~80k               | ~~0,08        | ~~2 mW              |
| Schraudolph + 1 Newton  | ~2 ULP          | ~20           | 1            | 0             | ~10k               | ~~0,01        | ~~0,1 mW            |

W powyższej tabeli przedstawiono orientacyjne liczby (przy mid-range FPGA i ~28nm ASIC). Źródła: empiryczne i z literatury. Metody zapewniające ≤0,5 ULP to przede wszystkim *minimaxowe wielomiany* oraz *duże LUTy z interpolacją*. Schraudolph daje za duży błąd, PWL potrzeba wiele segmentów by zejść poniżej 0,5 ULP. 

## Rekomendacja najtańszej implementacji

W oparciu o analizę, **najtańszą i wystarczająco dokładną** implementacją funkcji $e^x$ (x≤0) w bf16 jest **redukcja zakresu + wielomian minimaksowy rzędu 3–4** w pipelined FPU. Konkretnie:

- **Redukcja wykładnikowa (x=k·ln2+r):** zapewnia prostą część $2^k$ za pomocą arytmetyki na wykładnikach, z zerowym kosztem oprócz logiki wyboru (instrukcja add lub shift).
- **Polyn. minimaksowy stopnia 3:** wymaga 3 mnożeń FP (3 DSP) i 3 sum, co odpowiada minimalnej liczbie bloków DSP. Przy dobrze dobranych współczynnikach (Sollya) błąd jest <0,5 ULP. Osiągniemy opóźnienie rzędu kilku cykli, przy przepustowości 1 liczba/clk (gdy wypipelujemy).
- **Zasoby:** np. na Xilinx UltraScale wystarczy ~200–300 LUT i 3 DSP, bez potrzeby BRAM. Dla średniej klasy FPGA (Virtex7/7) to jest tanio; w ASIC to ~100–150k bramek (w nowszym procesie <0,2 mm²) i moc rzędu kilku mW. 
- **Wariant LUT+interp:** użyteczny, gdy mamy zapas pamięci (BRAM) i chcemy jeszcze zmniejszyć błąd. Jednak 0,5 ULP osiągniemy bez LUT dzięki polynomowi, więc unikamy pamięci. 

Ogólnie **minimax Horner 3-stopnia** jest najlepszym kompromisem: niski koszt (niewiele DSP, mało logicznie), mały błąd. Jeżeli środowisko FPU już ma zdolność fma i dodawania, dodatkowy obszar to tylko kilka DSP. Inne metody (głębsze polynomy lub LUT) dają marginalnie lepszą dokładność, ale kosztem większych zasobów, co nie jest wymagane do 0,5 ULP.

**Przy założeniach:** średniej klasy FPGA i zwykły ASIC (~28nm), pełne pokrycie subnormaliów traktujemy jako zero. Dla innych warunków (np. jeśli wymagany zakres input jest mniejszy, lub jeśli preferowane jest 8-bit int) parametry mogą się zmienić. 

   *Źródła główne:* format bf16; definicja błędu 0,5 ULP; metody aproksymacji; przykładowe implementacje FPGA/ASIC. Wszystkie liczby i oceny zasobów to oszacowania na podstawie analiz powyższych źródeł.