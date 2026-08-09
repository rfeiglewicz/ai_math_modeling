// Przepustowosc exp(x) w formacie BF16 na GPU.
//
// Punkt odniesienia dla rdzeni FPGA z docs/throughput_analysis.md.  Tam
// przepustowosc wychodzila z modelu N * Fmax; tutaj jest mierzona na krzemie.
//
// Co ten program mierzy:
//
//   1. Przepustowosc strumieniowa - wczytaj bf16, policz exp, zapisz bf16.
//      To jest odpowiednik tego, co robia rdzenie FPGA, i jedyna liczba
//      porownywalna 1:1.  Uwaga: przy 4 bajtach ruchu na jedna operacje
//      jest to zadanie **ograniczone przez pamiec**, a nie przez jednostki
//      obliczeniowe - dlatego mierzymy tez czysty kernel kopiujacy, zeby
//      bylo widac sufit.
//
//   2. Sufit obliczeniowy - lancuch zaleznych exp w rejestrach, bez ruchu
//      do pamieci.  Pokazuje, ile GPU umie policzyc, gdy dane sa juz na
//      miejscu.  To NIE jest przepustowosc uzyteczna, tylko gorne
//      ograniczenie sprzetu.
//
//   3. Blad ULP - wyczerpujaco, po wszystkich 65536 wzorcach bitowych BF16.
//      Projekt wymaga <= 0.5 ULP, wiec sprawdzamy, czy GPU to spelnia.
//
// Liczba probek jest parametryzowalna (--samples).  Ma to znaczenie, bo przy
// malym N dane siedza w L2 i przepustowosc jest kilkukrotnie wyzsza niz przy
// duzym N, gdy trzeba jechac do DRAM.  Opcja --sweep pokazuje ten prog.
//
// Kompilacja:
//   nvcc -O3 -arch=native -o bf16_exp_throughput bf16_exp_throughput.cu
// albo z katalogu ai_math_modeling:
//   make cuda_throughput

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Obsluga bledow
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err_ = (expr);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA blad %s:%d: %s\n  przy: %s\n",          \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #expr); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Konwersje BF16 <-> float po stronie hosta
//
// Robimy je recznie, zeby zaokraglanie bylo jawne i dalo sie je sprawdzic.
// BF16 to po prostu obciete gorne 16 bitow float32, z zaokragleniem do
// najblizszej parzystej (RNE) - dokladnie tak, jak w bf16_round.sv.
// ---------------------------------------------------------------------------

static inline float bf16_to_f32(uint16_t b)
{
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t f32_to_bf16_rne(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));

    // NaN: zachowaj NaN i ustaw bit quiet, zeby obciecie mantysy nie zamienilo
    // go przypadkiem w nieskonczonosc.
    if (((u >> 23) & 0xffu) == 0xffu && (u & 0x7fffffu) != 0u)
        return static_cast<uint16_t>((u >> 16) | 0x0040u);

    uint32_t lsb = (u >> 16) & 1u;          // bit, ktory zostanie
    u += 0x7fffu + lsb;                     // RNE: pol jednostki + korekta remisu
    return static_cast<uint16_t>(u >> 16);
}

static inline double bf16_to_double(uint16_t b)
{
    return static_cast<double>(bf16_to_f32(b));
}

// ---------------------------------------------------------------------------
// Generator liczb losowych oparty na liczniku
//
// Kazdy watek liczy swoja wartosc z indeksu i ziarna, bez stanu i bez
// dodatkowej biblioteki.  Wynik jest deterministyczny i identyczny niezaleznie
// od konfiguracji siatki, wiec pomiary sa powtarzalne.
// Mieszanie: finalizator z splitmix64.
// ---------------------------------------------------------------------------

__host__ __device__ static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// Rownomiernie z [lo, hi).
__host__ __device__ static inline float uniform_from(uint64_t idx, uint64_t seed,
                                                     float lo, float hi)
{
    uint64_t h = splitmix64(idx ^ (seed * 0x2545f4914f6cdd1dull));
    // 24 bity na mantyse float - dokladnie tyle, ile float uniesie bez strat.
    float unit = static_cast<float>(h >> 40) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * unit;
}

// ---------------------------------------------------------------------------
// Kernele
// ---------------------------------------------------------------------------

// Inicjalizacja pamieci wejsciowej losowymi wartosciami BF16.
__global__ void k_init_random(__nv_bfloat16 *__restrict__ dst, size_t n,
                              uint64_t seed, float lo, float hi)
{
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = __float2bfloat16(uniform_from(i, seed, lo, hi));
    }
}

// Wariant 1: skalarny, jedna wartosc BF16 na iteracje.
__global__ void k_exp_bf16_scalar(const __nv_bfloat16 *__restrict__ src,
                                  __nv_bfloat16 *__restrict__ dst, size_t n)
{
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = hexp(src[i]);
    }
}

// Wariant 2: para BF16 naraz (h2exp), dostep 32-bitowy.
__global__ void k_exp_bf16x2(const __nv_bfloat162 *__restrict__ src,
                             __nv_bfloat162 *__restrict__ dst, size_t n2)
{
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n2; i += stride) {
        dst[i] = h2exp(src[i]);
    }
}

// Wariant 3: osiem BF16 na watek przez dostep 128-bitowy (float4).
// Szeroki dostep lepiej wysyca kontroler pamieci.
__global__ void k_exp_bf16x8(const float4 *__restrict__ src,
                             float4 *__restrict__ dst, size_t n8)
{
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n8; i += stride) {
        float4 raw = src[i];
        __nv_bfloat162 v[4];
        std::memcpy(v, &raw, sizeof(v));
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = h2exp(v[j]);
        std::memcpy(&raw, v, sizeof(raw));
        dst[i] = raw;
    }
}

// Wariant 4: BF16 przez float32 i expf - tak wyglada kod, ktory nie uzywa
// wprost intrinsika BF16.  Do porownania dokladnosci i szybkosci.
__global__ void k_exp_bf16_via_expf(const __nv_bfloat16 *__restrict__ src,
                                    __nv_bfloat16 *__restrict__ dst, size_t n)
{
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = __float2bfloat16(expf(__bfloat162float(src[i])));
    }
}

// Wariant 5: BF16 przez szybkie __expf (sprzetowe MUFU.EX2, mniej dokladne).
__global__ void k_exp_bf16_via_fastexpf(const __nv_bfloat16 *__restrict__ src,
                                        __nv_bfloat16 *__restrict__ dst, size_t n)
{
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = __float2bfloat16(__expf(__bfloat162float(src[i])));
    }
}

// Odniesienie: czysta kopia, bez liczenia.  Wyznacza sufit narzucony przez
// pamiec - zadny kernel liczacy exp przy tym samym ruchu danych nie moze byc
// szybszy.
__global__ void k_copy(const float4 *__restrict__ src, float4 *__restrict__ dst,
                       size_t n8)
{
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n8; i += stride) {
        dst[i] = src[i];
    }
}

// Sufit obliczeniowy: lancuch zaleznych exp w rejestrach.
//
// Odwzorowanie v -> exp(v - 1) jest stabilne: dla v w (0, 1] argument wpada w
// (-1, 0], a exp z tego wraca do (e^-1, 1].  Wartosci nie uciekaja do zera ani
// do nieskonczonosci, a kazda iteracja zalezy od poprzedniej, wiec kompilator
// nie moze niczego wyniesc przed petle ani zwektoryzowac.
//
// Uwaga przy interpretacji: na iteracje przypada exp ORAZ odejmowanie, wiec
// wynik jest dolnym oszacowaniem czystej przepustowosci exp.
__global__ void k_exp_chain(const __nv_bfloat16 *__restrict__ src,
                            __nv_bfloat16 *__restrict__ dst, size_t n,
                            int chain)
{
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const __nv_bfloat16 one = __float2bfloat16(1.0f);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        __nv_bfloat16 v = src[i];
        for (int k = 0; k < chain; ++k)
            v = hexp(__hsub(v, one));
        dst[i] = v;   // zapis konieczny, inaczej kompilator usunie cala petle
    }
}

// ---------------------------------------------------------------------------
// Odczyt biezacych zegarow przez NVML
//
// To nie jest ozdobnik. GPU w laptopie startuje w stanie oszczedzania (P8,
// pamiec na ~13% zegara) i wchodzi na pelne zegary dopiero po kilku sekundach
// ciaglego obciazenia. Pomiar zrobiony za wczesnie zanizy wynik kilkukrotnie,
// wiec raportujemy stan ukladu razem z wynikiem, zeby dalo sie to wychwycic.
//
// NVML ladujemy przez dlopen: biblioteka przychodzi ze sterownikiem, ale nie
// chcemy twardej zaleznosci przy linkowaniu. Gdy jej nie ma, program dziala
// dalej, tylko bez informacji o zegarach.
// ---------------------------------------------------------------------------

struct GpuClocks {
    bool valid = false;
    int pstate = -1;
    unsigned sm_mhz = 0;
    unsigned mem_mhz = 0;
};

class Nvml {
public:
    void open(const cudaDeviceProp &prop)
    {
        lib_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib_) lib_ = dlopen("libnvidia-ml.so", RTLD_LAZY);
        if (!lib_) return;

        auto sym = [&](const char *n) { return dlsym(lib_, n); };
        init_     = reinterpret_cast<int (*)()>(sym("nvmlInit_v2"));
        shutdown_ = reinterpret_cast<int (*)()>(sym("nvmlShutdown"));
        by_pci_   = reinterpret_cast<int (*)(const char *, void **)>(
            sym("nvmlDeviceGetHandleByPciBusId_v2"));
        clock_    = reinterpret_cast<int (*)(void *, int, unsigned *)>(
            sym("nvmlDeviceGetClockInfo"));
        pstate_   = reinterpret_cast<int (*)(void *, int *)>(
            sym("nvmlDeviceGetPerformanceState"));

        if (!init_ || !by_pci_ || !clock_ || !pstate_) return;
        if (init_() != 0) return;

        // Dopasowanie po adresie PCI, a nie po indeksie: numeracja NVML i CUDA
        // nie musi byc taka sama.
        char bus[32];
        std::snprintf(bus, sizeof(bus), "%08x:%02x:%02x.0", prop.pciDomainID,
                      prop.pciBusID, prop.pciDeviceID);
        if (by_pci_(bus, &dev_) != 0) dev_ = nullptr;
        ok_ = (dev_ != nullptr);
    }

    GpuClocks read() const
    {
        GpuClocks c;
        if (!ok_) return c;
        int ps = -1;
        unsigned sm = 0, mem = 0;
        if (pstate_(dev_, &ps) != 0) return c;
        if (clock_(dev_, 1 /* NVML_CLOCK_SM */, &sm) != 0) return c;
        if (clock_(dev_, 2 /* NVML_CLOCK_MEM */, &mem) != 0) return c;
        c.valid = true;
        c.pstate = ps;
        c.sm_mhz = sm;
        c.mem_mhz = mem;
        return c;
    }

    ~Nvml()
    {
        if (ok_ && shutdown_) shutdown_();
        if (lib_) dlclose(lib_);
    }

private:
    void *lib_ = nullptr;
    void *dev_ = nullptr;
    bool ok_ = false;
    int (*init_)() = nullptr;
    int (*shutdown_)() = nullptr;
    int (*by_pci_)(const char *, void **) = nullptr;
    int (*clock_)(void *, int, unsigned *) = nullptr;
    int (*pstate_)(void *, int *) = nullptr;
};

static Nvml g_nvml;

// ---------------------------------------------------------------------------
// Pomiar czasu
// ---------------------------------------------------------------------------

struct Timing {
    double best_ms;
    double median_ms;
    GpuClocks clocks;   // stan ukladu tuz po serii pomiarowej
};

template <typename F>
static Timing time_kernel(F &&launch, int repeats, double warmup_ms)
{
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    // Rozgrzewka liczona CZASEM, nie liczba iteracji. Liczba iteracji nie
    // wystarcza, bo krotki kernel moze sie wykonac tysiace razy, zanim uklad
    // w ogole zauwazy obciazenie i podniesie zegary.
    {
        auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            double el = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            if (el >= warmup_ms) break;
        }
    }

    std::vector<double> ms;
    ms.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventRecord(beg));
        launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float dt = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&dt, beg, end));
        ms.push_back(static_cast<double>(dt));
    }
    CUDA_CHECK(cudaGetLastError());

    Timing t;
    t.clocks = g_nvml.read();
    std::sort(ms.begin(), ms.end());
    t.best_ms = ms.front();
    t.median_ms = ms[ms.size() / 2];

    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return t;
}

// ---------------------------------------------------------------------------
// Konfiguracja siatki
//
// Zamiast zgadywac liczbe blokow pytamy sterownik, ile blokow na SM da sie
// uruchomic jednoczesnie, i wypelniamy uklad dokladnie tyle razy.  Petla
// grid-stride w kazdym kernelu obsluzy dowolne N.
// ---------------------------------------------------------------------------

template <typename K>
static int grid_for(K kernel, int block, int sm_count)
{
    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                             block, 0));
    if (per_sm < 1) per_sm = 1;
    return per_sm * sm_count;
}

// ---------------------------------------------------------------------------
// Raport przepustowosci
// ---------------------------------------------------------------------------

struct Result {
    const char *name;
    double gops;        // miliardy elementow na sekunde
    double gbps;        // efektywny ruch do/z pamieci
    double ms;
    GpuClocks clocks;
};

static void print_header()
{
    std::printf("\n%-28s %12s %12s %12s %8s\n", "wariant", "czas [ms]",
                "[Gop/s]", "[GB/s]", "pstate");
    std::printf("%s\n", std::string(78, '-').c_str());
}

static void print_row(const Result &r)
{
    char ps[16] = "?";
    if (r.clocks.valid) std::snprintf(ps, sizeof(ps), "P%d", r.clocks.pstate);
    std::printf("%-28s %12.4f %12.2f %12.1f %8s\n", r.name, r.ms, r.gops,
                r.gbps, ps);
}

// ---------------------------------------------------------------------------
// Wyczerpujace sprawdzenie bledu ULP
//
// BF16 ma 16 bitow, czyli tylko 65536 mozliwych wejsc.  Mozna wiec sprawdzic
// wszystkie, bez probkowania i bez statystyki.  Referencja: exp() w double,
// potem poprawne zaokraglenie do BF16 przez porownanie sasiadow (nie przez
// double -> float -> bf16, bo to daloby podwojne zaokraglenie).
// ---------------------------------------------------------------------------

// Rozmiar ULP BF16 w okolicy wartosci v.
static double bf16_ulp_at(double v)
{
    double a = std::fabs(v);
    if (a == 0.0 || !std::isfinite(a))
        return std::ldexp(1.0, -133);       // najmniejszy subnormalny BF16
    int e;
    std::frexp(a, &e);                      // a w [0.5, 1) * 2^e
    int ulp_exp = (e - 1) - 7;              // 7 bitow mantysy zapisanych
    if (ulp_exp < -133) ulp_exp = -133;
    return std::ldexp(1.0, ulp_exp);
}

// Poprawnie zaokraglona wartosc exp(x) w BF16.
static uint16_t reference_exp_bf16(uint16_t xb, double *exact_out)
{
    double x = bf16_to_double(xb);
    double e = std::exp(x);
    *exact_out = e;

    if (std::isnan(x)) return 0x7fc0;

    // Kandydat startowy, potem sprawdzamy jego sasiadow w dziedzinie BF16 i
    // wybieramy tego, ktory jest naprawde najblizej. To omija podwojne
    // zaokraglenie double -> float -> bf16.
    uint16_t start = f32_to_bf16_rne(static_cast<float>(e));
    uint16_t best = start;
    double best_d = std::fabs(bf16_to_double(start) - e);

    for (int off = -2; off <= 2; ++off) {
        int cand = static_cast<int>(start) + off;
        if (cand < 0 || cand > 0xffff) continue;
        uint16_t c = static_cast<uint16_t>(cand);
        double cv = bf16_to_double(c);
        if (!std::isfinite(cv) && std::isfinite(e)) continue;
        double d = std::fabs(cv - e);
        // Remis rozstrzygamy na korzysc parzystej mantysy (RNE).
        if (d < best_d || (d == best_d && (c & 1u) == 0u)) {
            best_d = d;
            best = c;
        }
    }
    return best;
}

struct UlpStats {
    double max_ulp;
    uint16_t worst_in;
    uint16_t worst_got;
    uint16_t worst_ref;
    long long over_half;
    long long checked;
};

static UlpStats ulp_check(const std::vector<uint16_t> &got, bool only_nonpositive)
{
    UlpStats s{};
    s.max_ulp = 0.0;
    for (uint32_t b = 0; b < 65536u; ++b) {
        uint16_t xb = static_cast<uint16_t>(b);
        double x = bf16_to_double(xb);
        if (!std::isfinite(x)) continue;                 // NaN/Inf osobno
        if (only_nonpositive && !(x <= 0.0)) continue;   // dziedzina projektu

        double exact = 0.0;
        uint16_t ref = reference_exp_bf16(xb, &exact);
        if (!std::isfinite(exact)) continue;             // przepelnienie

        double gv = bf16_to_double(got[b]);
        if (!std::isfinite(gv)) continue;

        double err = std::fabs(gv - exact) / bf16_ulp_at(exact);
        s.checked++;
        if (err > 0.5 + 1e-9) s.over_half++;
        if (err > s.max_ulp) {
            s.max_ulp = err;
            s.worst_in = xb;
            s.worst_got = got[b];
            s.worst_ref = ref;
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Argumenty
// ---------------------------------------------------------------------------

struct Options {
    size_t samples = 1ull << 26;   // 67.1 M probek = 128 MiB wej + 128 MiB wyj
    int repeats = 20;
    int chain = 128;
    uint64_t seed = 12345;
    float lo = -20.0f;             // dziedzina softmax po odjeciu maksimum
    float hi = 0.0f;
    int block = 256;
    double warmup_ms = 2000.0;     // patrz komentarz przy time_kernel
    bool sweep = false;
    bool check = true;
    bool csv = false;
};

static void usage(const char *prog)
{
    std::printf(
        "Uzycie: %s [opcje]\n\n"
        "  -n, --samples N     liczba probek BF16 (domyslnie 67108864 = 2^26)\n"
        "                      przyrostki k/M/G dzialaja, np. --samples 256M\n"
        "  -r, --repeats R     ile razy powtorzyc pomiar (domyslnie 20)\n"
        "      --chain K       dlugosc lancucha exp dla sufitu obliczeniowego\n"
        "                      (domyslnie 128)\n"
        "      --seed S        ziarno generatora (domyslnie 12345)\n"
        "      --lo X          dolna granica losowanych wartosci (domyslnie -20)\n"
        "      --hi X          gorna granica losowanych wartosci (domyslnie 0)\n"
        "      --block B       rozmiar bloku watkow (domyslnie 256)\n"
        "      --warmup MS     dlugosc rozgrzewki na wariant (domyslnie 2000 ms)\n"
        "                      GPU w laptopie potrzebuje kilku sekund ciaglego\n"
        "                      obciazenia, zeby wejsc w stan P0; za krotka\n"
        "                      rozgrzewka zanizy wynik nawet kilkukrotnie\n"
        "      --sweep         przemiataj N od 64 Ki do zadanego, pokaz prog\n"
        "                      miedzy L2 a DRAM\n"
        "      --no-check      pomin wyczerpujace sprawdzenie bledu ULP\n"
        "      --csv           dodatkowo wypisz wyniki w formacie CSV\n"
        "  -h, --help          ta pomoc\n\n"
        "Dziedzina domyslna to x <= 0, bo taki jest zakres w softmaksie po\n"
        "odjeciu maksimum (stabilizacja numeryczna).\n",
        prog);
}

static size_t parse_size(const char *s)
{
    char *endp = nullptr;
    double v = std::strtod(s, &endp);
    if (endp && *endp) {
        switch (*endp) {
            case 'k': case 'K': v *= 1024.0; break;
            case 'm': case 'M': v *= 1024.0 * 1024.0; break;
            case 'g': case 'G': v *= 1024.0 * 1024.0 * 1024.0; break;
            default: break;
        }
    }
    if (v < 1.0) v = 1.0;
    return static_cast<size_t>(v);
}

static bool parse_args(int argc, char **argv, Options *o)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "brak wartosci dla %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(argv[0]); return false; }
        else if (a == "-n" || a == "--samples") o->samples = parse_size(next("--samples"));
        else if (a == "-r" || a == "--repeats") o->repeats = std::atoi(next("--repeats"));
        else if (a == "--chain")   o->chain   = std::atoi(next("--chain"));
        else if (a == "--seed")    o->seed    = std::strtoull(next("--seed"), nullptr, 10);
        else if (a == "--lo")      o->lo      = std::strtof(next("--lo"), nullptr);
        else if (a == "--hi")      o->hi      = std::strtof(next("--hi"), nullptr);
        else if (a == "--block")   o->block   = std::atoi(next("--block"));
        else if (a == "--warmup")  o->warmup_ms = std::strtod(next("--warmup"), nullptr);
        else if (a == "--sweep")   o->sweep   = true;
        else if (a == "--no-check") o->check  = false;
        else if (a == "--csv")     o->csv     = true;
        else {
            std::fprintf(stderr, "nieznana opcja: %s (--help po liste)\n", a.c_str());
            std::exit(2);
        }
    }
    if (o->repeats < 1) o->repeats = 1;
    if (o->chain < 1) o->chain = 1;
    // Zaokraglij w gore do wielokrotnosci 8, zeby wariant 128-bitowy dostal
    // pelne paczki i wszystkie warianty liczyly dokladnie tyle samo elementow.
    o->samples = (o->samples + 7u) & ~static_cast<size_t>(7u);
    return true;
}

// ---------------------------------------------------------------------------
// Pojedynczy przebieg pomiarowy dla zadanego N
// ---------------------------------------------------------------------------

static std::vector<Result> run_once(const Options &o, size_t n,
                                    __nv_bfloat16 *d_in, __nv_bfloat16 *d_out,
                                    int sm_count, bool with_chain)
{
    const int block = o.block;
    const size_t n2 = n / 2;
    const size_t n8 = n / 8;
    const double bytes = static_cast<double>(n) * 4.0;   // 2 B wej + 2 B wyj

    std::vector<Result> out;
    auto add = [&](const char *name, const Timing &t, double elems, double b) {
        Result r;
        r.name = name;
        r.ms = t.best_ms;
        r.gops = elems / (t.best_ms * 1e-3) / 1e9;
        r.gbps = b / (t.best_ms * 1e-3) / 1e9;
        r.clocks = t.clocks;
        out.push_back(r);
    };

    {
        int grid = grid_for(k_copy, block, sm_count);
        Timing t = time_kernel([&] {
            k_copy<<<grid, block>>>(reinterpret_cast<const float4 *>(d_in),
                                    reinterpret_cast<float4 *>(d_out), n8);
        }, o.repeats, o.warmup_ms);
        add("kopia (sufit pamieci)", t, static_cast<double>(n), bytes);
    }
    {
        int grid = grid_for(k_exp_bf16_scalar, block, sm_count);
        Timing t = time_kernel([&] {
            k_exp_bf16_scalar<<<grid, block>>>(d_in, d_out, n);
        }, o.repeats, o.warmup_ms);
        add("hexp skalarny", t, static_cast<double>(n), bytes);
    }
    {
        int grid = grid_for(k_exp_bf16x2, block, sm_count);
        Timing t = time_kernel([&] {
            k_exp_bf16x2<<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat162 *>(d_in),
                reinterpret_cast<__nv_bfloat162 *>(d_out), n2);
        }, o.repeats, o.warmup_ms);
        add("h2exp (2 na watek)", t, static_cast<double>(n), bytes);
    }
    {
        int grid = grid_for(k_exp_bf16x8, block, sm_count);
        Timing t = time_kernel([&] {
            k_exp_bf16x8<<<grid, block>>>(
                reinterpret_cast<const float4 *>(d_in),
                reinterpret_cast<float4 *>(d_out), n8);
        }, o.repeats, o.warmup_ms);
        add("h2exp x4 (128-bit)", t, static_cast<double>(n), bytes);
    }
    {
        int grid = grid_for(k_exp_bf16_via_expf, block, sm_count);
        Timing t = time_kernel([&] {
            k_exp_bf16_via_expf<<<grid, block>>>(d_in, d_out, n);
        }, o.repeats, o.warmup_ms);
        add("expf przez float32", t, static_cast<double>(n), bytes);
    }
    {
        int grid = grid_for(k_exp_bf16_via_fastexpf, block, sm_count);
        Timing t = time_kernel([&] {
            k_exp_bf16_via_fastexpf<<<grid, block>>>(d_in, d_out, n);
        }, o.repeats, o.warmup_ms);
        add("__expf przez float32", t, static_cast<double>(n), bytes);
    }

    if (with_chain) {
        int grid = grid_for(k_exp_chain, block, sm_count);
        Timing t = time_kernel([&] {
            k_exp_chain<<<grid, block>>>(d_in, d_out, n, o.chain);
        }, o.repeats, o.warmup_ms);
        // Ruch do pamieci jest ten sam, ale operacji jest chain razy wiecej.
        Result r;
        r.name = "lancuch exp (sufit obl.)";
        r.ms = t.best_ms;
        r.gops = static_cast<double>(n) * o.chain / (t.best_ms * 1e-3) / 1e9;
        r.gbps = bytes / (t.best_ms * 1e-3) / 1e9;
        r.clocks = t.clocks;
        out.push_back(r);
    }
    return out;
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    Options o;
    if (!parse_args(argc, argv, &o)) return 0;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    if (prop.major < 8) {
        std::fprintf(stderr,
                     "Ten program wymaga architektury sm_80 lub nowszej "
                     "(intrinsiki BF16). Wykryto sm_%d%d.\n",
                     prop.major, prop.minor);
        return 1;
    }

    // Teoretyczna przepustowosc pamieci: szyna DDR, stad czynnik 2.
    const double peak_gbps =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;

    g_nvml.open(prop);
    GpuClocks idle = g_nvml.read();

    std::printf("GPU            : %s (sm_%d%d, %d SM)\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("Pamiec         : %.1f GiB, szyna %d-bit, teoretycznie %.1f GB/s\n",
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                prop.memoryBusWidth, peak_gbps);
    std::printf("Cache L2       : %.1f MiB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    if (idle.valid) {
        std::printf("Zegary teraz   : P%d, SM %u MHz, pamiec %u MHz "
                    "(maksimum %d MHz)\n",
                    idle.pstate, idle.sm_mhz, idle.mem_mhz,
                    prop.memoryClockRate / 1000);
    }

    const size_t n = o.samples;
    const size_t bytes_in = n * sizeof(__nv_bfloat16);

    std::printf("Probek         : %zu (%.2f M), bufory 2 x %.1f MiB\n", n,
                n / 1e6, bytes_in / (1024.0 * 1024.0));
    std::printf("Zakres losowy  : [%.3g, %.3g)%s\n", o.lo, o.hi,
                (o.hi <= 0.0f) ? "   (dziedzina softmax: x <= 0)" : "");
    std::printf("Ziarno         : %llu\n",
                static_cast<unsigned long long>(o.seed));
    std::printf("Rozgrzewka     : %.0f ms na wariant\n", o.warmup_ms);

    __nv_bfloat16 *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes_in));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_in));

    // --- inicjalizacja pamieci losowymi wartosciami ---
    {
        int grid = grid_for(k_init_random, o.block, prop.multiProcessorCount);
        cudaEvent_t b, e;
        CUDA_CHECK(cudaEventCreate(&b));
        CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(b));
        k_init_random<<<grid, o.block>>>(d_in, n, o.seed, o.lo, o.hi);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, b, e));
        std::printf("Inicjalizacja  : %.2f ms\n", ms);
        CUDA_CHECK(cudaEventDestroy(b));
        CUDA_CHECK(cudaEventDestroy(e));

        // Kontrola, ze w buforze naprawde sa rozne wartosci, a nie same zera.
        size_t probe = std::min<size_t>(n, 4096);
        std::vector<uint16_t> head(probe);
        CUDA_CHECK(cudaMemcpy(head.data(), d_in, probe * 2, cudaMemcpyDeviceToHost));
        double sum = 0.0, mn = 1e30, mx = -1e30;
        size_t distinct = 0;
        std::vector<uint16_t> uniq(head);
        std::sort(uniq.begin(), uniq.end());
        distinct = static_cast<size_t>(std::unique(uniq.begin(), uniq.end()) - uniq.begin());
        for (uint16_t v : head) {
            double f = bf16_to_double(v);
            sum += f;
            mn = std::min(mn, f);
            mx = std::max(mx, f);
        }
        std::printf("Kontrola danych: %zu prob, %zu roznych, min %.3f, "
                    "max %.3f, srednia %.3f\n",
                    probe, distinct, mn, mx, sum / probe);
    }

    // --- pomiar ---
    std::printf("\n=== Przepustowosc, N = %.2f M probek ===\n", n / 1e6);
    print_header();
    std::vector<Result> res =
        run_once(o, n, d_in, d_out, prop.multiProcessorCount, true);
    for (const Result &r : res) print_row(r);

    // Najlepszy wariant strumieniowy (pomijamy kopie i lancuch).
    const Result *best = nullptr;
    for (size_t i = 1; i + 1 < res.size(); ++i)
        if (!best || res[i].gops > best->gops) best = &res[i];
    const Result &copy = res.front();
    const Result &chain = res.back();

    std::printf("\nNajszybszy wariant strumieniowy: %s, %.2f Gop/s (%.1f GB/s, "
                "%.0f%% teoretycznej przepustowosci pamieci).\n",
                best->name, best->gops, best->gbps, 100.0 * best->gbps / peak_gbps);
    std::printf("Kernel kopiujacy, bez zadnych obliczen, osiaga %.2f Gop/s przy "
                "tym samym ruchu danych.\n", copy.gops);
    if (best->gops > 0.0) {
        double overhead = copy.gops / best->gops - 1.0;
        if (overhead < 0.02)
            std::printf("Roznica jest ponizej 2%%, czyli exp jest **darmowy**: "
                        "uklad i tak czeka na pamiec.\n");
        else
            std::printf("Roznica %.0f%% to koszt samego liczenia exp.\n",
                        100.0 * overhead);
    }
    std::printf("Sufit obliczeniowy (dane w rejestrach): %.1f Gop/s, czyli %.0fx "
                "wiecej niz wariant strumieniowy.\n",
                chain.gops, chain.gops / best->gops);
    std::printf("Wniosek: przy 4 bajtach ruchu na operacje exp w BF16 jest "
                "zadaniem ograniczonym przez pamiec, a nie przez ALU.\n");

    // Ostrzezenie o pomiarze przy zanizonych zegarach - bez tego latwo podac
    // wynik kilkukrotnie za niski i nawet tego nie zauwazyc.
    if (best->clocks.valid) {
        int mem_max_mhz = prop.memoryClockRate / 1000;
        double ratio = mem_max_mhz > 0
                           ? static_cast<double>(best->clocks.mem_mhz) / mem_max_mhz
                           : 1.0;
        if (best->clocks.pstate != 0 || ratio < 0.9) {
            std::printf("\nUWAGA: podczas pomiaru uklad byl w stanie P%d, pamiec "
                        "na %u z %d MHz (%.0f%%).\n"
                        "Wyniki sa zanizone. Wydluz rozgrzewke (--warmup) albo "
                        "zwieksz N,\n"
                        "zeby obciazenie bylo ciagle wystarczajaco dlugo.\n",
                        best->clocks.pstate, best->clocks.mem_mhz, mem_max_mhz,
                        100.0 * ratio);
        } else {
            std::printf("\nUklad byl podczas pomiaru w P%d z pamiecia na %u MHz "
                        "(%.0f%% maksimum), wiec pomiar jest miarodajny.\n",
                        best->clocks.pstate, best->clocks.mem_mhz, 100.0 * ratio);
        }
    }

    // --- przemiatanie N ---
    if (o.sweep) {
        std::printf("\n=== Przemiatanie N (najlepszy wariant strumieniowy) ===\n");
        std::printf("\n%14s %10s %10s %10s %10s %8s\n", "N [probek]",
                    "razem[MiB]", "czas [us]", "[Gop/s]", "[GB/s]", "gdzie");
        std::printf("%s\n", std::string(68, '-').c_str());
        // Uklad jest juz w P0 po pomiarze glownym i pozostanie tam przy ciaglym
        // obciazeniu, wiec krotsza rozgrzewka wystarczy.
        Options os = o;
        os.warmup_ms = std::max(250.0, o.warmup_ms / 8.0);
        double t_min_us = 0.0;
        for (size_t m = 1ull << 16; m <= n; m <<= 1) {
            size_t mm = (m + 7u) & ~static_cast<size_t>(7u);
            std::vector<Result> r =
                run_once(os, mm, d_in, d_out, prop.multiProcessorCount, false);
            const Result *b = nullptr;
            for (size_t i = 1; i < r.size(); ++i)
                if (!b || r[i].gops > b->gops) b = &r[i];
            // Oba bufory razem, bo obie tablice walcza o to samo L2.
            double mib = 2.0 * mm * 2.0 / (1024.0 * 1024.0);
            // "<=", bo przy dokladnie rownym rozmiarze dane wciaz sie mieszcza.
            const char *where =
                (2.0 * mm * 2.0 <= prop.l2CacheSize) ? "L2" : "DRAM";
            double us = b->ms * 1000.0;
            if (t_min_us == 0.0 || us < t_min_us) t_min_us = us;
            std::printf("%14zu %10.2f %10.2f %10.2f %10.1f %8s\n", mm, mib, us,
                        b->gops, b->gbps, where);
        }
        std::printf("\nSpadek przy przejsciu z L2 do DRAM pokazuje, ze o wyniku "
                    "decyduje pamiec, a nie jednostki obliczeniowe.\n");
        std::printf("Przy najmniejszych N wynik zaniza narzut uruchomienia "
                    "kernela: najkrotszy\nzmierzony czas to %.2f us i ponizej "
                    "tego progu nie da sie zejsc niezaleznie od N.\n", t_min_us);
    }

    // --- wyczerpujace sprawdzenie ULP ---
    if (o.check) {
        std::printf("\n=== Blad ULP, wszystkie 65536 wzorcow bitowych BF16 ===\n");

        std::vector<uint16_t> all(65536);
        for (uint32_t b = 0; b < 65536u; ++b) all[b] = static_cast<uint16_t>(b);

        __nv_bfloat16 *d_all = nullptr, *d_res = nullptr;
        CUDA_CHECK(cudaMalloc(&d_all, 65536 * 2));
        CUDA_CHECK(cudaMalloc(&d_res, 65536 * 2));
        CUDA_CHECK(cudaMemcpy(d_all, all.data(), 65536 * 2, cudaMemcpyHostToDevice));

        struct Variant {
            const char *name;
            void (*fn)(const __nv_bfloat16 *, __nv_bfloat16 *, size_t);
        };
        const Variant variants[] = {
            {"hexp",                 k_exp_bf16_scalar},
            {"expf przez float32",   k_exp_bf16_via_expf},
            {"__expf przez float32", k_exp_bf16_via_fastexpf},
        };

        std::printf("\n%-24s %10s %12s %14s %14s\n", "wariant", "zakres",
                    "max ULP", "> 0.5 ULP", "sprawdzonych");
        std::printf("%s\n", std::string(78, '-').c_str());

        std::vector<uint16_t> got(65536);
        for (const Variant &v : variants) {
            v.fn<<<256, 256>>>(d_all, d_res, 65536);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(got.data(), d_res, 65536 * 2,
                                  cudaMemcpyDeviceToHost));

            UlpStats sn = ulp_check(got, true);    // x <= 0
            UlpStats sa = ulp_check(got, false);   // wszystkie skonczone
            std::printf("%-24s %10s %12.4f %14lld %14lld\n", v.name, "x <= 0",
                        sn.max_ulp, sn.over_half, sn.checked);
            std::printf("%-24s %10s %12.4f %14lld %14lld\n", "", "wszystkie",
                        sa.max_ulp, sa.over_half, sa.checked);
            if (sn.max_ulp > 0.5) {
                std::printf("%-24s   najgorszy dla x <= 0: x = %.8g, "
                            "otrzymano %.9g, poprawnie %.9g\n", "",
                            bf16_to_double(sn.worst_in),
                            bf16_to_double(sn.worst_got),
                            bf16_to_double(sn.worst_ref));
            }
        }

        std::printf("\nProjekt wymaga bledu <= 0.5 ULP w BF16 dla x <= 0, czyli\n"
                    "wyniku zaokraglonego poprawnie. Kolumna \"> 0.5 ULP\" mowi,\n"
                    "ile wejsc tego nie spelnia; zero oznacza, ze wariant jest\n"
                    "poprawnie zaokraglony na calej dziedzinie.\n");

        CUDA_CHECK(cudaFree(d_all));
        CUDA_CHECK(cudaFree(d_res));
    }

    if (o.csv) {
        std::printf("\nCSV\nwariant,czas_ms,gops,gbps\n");
        for (const Result &r : res)
            std::printf("%s,%.6f,%.4f,%.2f\n", r.name, r.ms, r.gops, r.gbps);
    }

    std::printf("\nUwaga do porownania z FPGA (docs/throughput_analysis.md):\n"
                "liczby GPU sa zmierzone na krzemie, liczby FPGA pochodza z\n"
                "syntezy bez implementacji i zakladaja 100%% wysycenia ukladu.\n"
                "Porownywalny jest wariant strumieniowy, bo rdzenie FPGA tez\n"
                "czytaja i zapisuja po jednej wartosci BF16 na operacje.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
