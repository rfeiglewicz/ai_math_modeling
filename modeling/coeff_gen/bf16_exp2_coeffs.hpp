#ifndef BF16_EXP2_COEFFS_HPP
#define BF16_EXP2_COEFFS_HPP

#include <cstdint>

// =============================================================
// exp2(x) Linear Approximation Coefficients for BF16
// =============================================================
//
// Generated: 2025-12-23 16:56:04
// Interval:  [-0.99609375, -0.001953125]
// LUT Size:  64
//
// Approximation: y = a * x + b
//
// Error Metrics (BF16 ULP):
//   Worst-case: 0.499632 ULP
//   Average:    0.250327 ULP
// =============================================================

namespace bf16_exp2 {

constexpr int LUT_SIZE = 64;
constexpr float INTERVAL_START = -0.99609375f;
constexpr float INTERVAL_END = -0.001953125f;

// Slope coefficients (a)
static const float coeffs_a[LUT_SIZE] = {
     3.4892838413e-01f,  // Idx 0 ulp=0.3938
     3.5272796512e-01f,  // Idx 1 ulp=0.4492
     3.5656892083e-01f,  // Idx 2 ulp=0.4869
     3.6045170180e-01f,  // Idx 3 ulp=0.4143
     3.6437676349e-01f,  // Idx 4 ulp=0.3928
     3.6834456630e-01f,  // Idx 5 ulp=0.4889
     3.7235557565e-01f,  // Idx 6 ulp=0.3357
     3.7641026202e-01f,  // Idx 7 ulp=0.4558
     3.8050910104e-01f,  // Idx 8 ulp=0.3429
     3.8465257349e-01f,  // Idx 9 ulp=0.4875
     3.8884116539e-01f,  // Idx 10 ulp=0.4160
     3.9307536807e-01f,  // Idx 11 ulp=0.4143
     3.9735567819e-01f,  // Idx 12 ulp=0.4436
     4.0168259783e-01f,  // Idx 13 ulp=0.4460
     4.0605663454e-01f,  // Idx 14 ulp=0.4212
     4.1047830139e-01f,  // Idx 15 ulp=0.3964
     4.1494811703e-01f,  // Idx 16 ulp=0.4600
     4.1946660578e-01f,  // Idx 17 ulp=0.4483
     4.2403429764e-01f,  // Idx 18 ulp=0.3963
     4.2865172840e-01f,  // Idx 19 ulp=0.4720
     4.3331943969e-01f,  // Idx 20 ulp=0.3888
     4.3803797902e-01f,  // Idx 21 ulp=0.4387
     4.4280789987e-01f,  // Idx 22 ulp=0.3205
     4.4762976176e-01f,  // Idx 23 ulp=0.4534
     4.5250413028e-01f,  // Idx 24 ulp=0.4455
     4.5743157719e-01f,  // Idx 25 ulp=0.3697
     4.6241268049e-01f,  // Idx 26 ulp=0.4774
     4.6744802444e-01f,  // Idx 27 ulp=0.4085
     4.7253819969e-01f,  // Idx 28 ulp=0.2860
     4.7768380332e-01f,  // Idx 29 ulp=0.3385
     4.8288543889e-01f,  // Idx 30 ulp=0.4148
     4.8814371657e-01f,  // Idx 31 ulp=0.4701
     4.9312550099e-01f,  // Idx 32 ulp=0.4957
     4.9849528563e-01f,  // Idx 33 ulp=0.4866
     5.0392354339e-01f,  // Idx 34 ulp=0.4918
     5.0941091100e-01f,  // Idx 35 ulp=0.4955
     5.1495803213e-01f,  // Idx 36 ulp=0.4537
     5.2056555745e-01f,  // Idx 37 ulp=0.3901
     5.2623414471e-01f,  // Idx 38 ulp=0.4851
     5.3196445884e-01f,  // Idx 39 ulp=0.4589
     5.3775717201e-01f,  // Idx 40 ulp=0.4749
     5.4361296368e-01f,  // Idx 41 ulp=0.4526
     5.4916028976e-01f,  // Idx 42 ulp=0.4535
     5.5476498639e-01f,  // Idx 43 ulp=0.4751
     5.6080598102e-01f,  // Idx 44 ulp=0.4571
     5.6691275777e-01f,  // Idx 45 ulp=0.4810
     5.7308603295e-01f,  // Idx 46 ulp=0.4860
     5.7932653070e-01f,  // Idx 47 ulp=0.4720
     5.8583327214e-01f,  // Idx 48 ulp=0.4664
     5.9802436898e-01f,  // Idx 49 ulp=0.4992
     5.9866135039e-01f,  // Idx 50 ulp=0.4774
     6.0518034508e-01f,  // Idx 51 ulp=0.4961
     6.1177032697e-01f,  // Idx 52 ulp=0.4804
     6.1822257613e-01f,  // Idx 53 ulp=0.4821
     6.2474331988e-01f,  // Idx 54 ulp=0.4984
     6.3154632860e-01f,  // Idx 55 ulp=0.4852
     6.3861193081e-01f,  // Idx 56 ulp=0.4925
     6.4548462852e-01f,  // Idx 57 ulp=0.4878
     6.5240296508e-01f,  // Idx 58 ulp=0.4772
     6.5939569064e-01f,  // Idx 59 ulp=0.4943
     6.6675802936e-01f,  // Idx 60 ulp=0.4807
     6.7383451821e-01f,  // Idx 61 ulp=0.4991
     6.8126353450e-01f,  // Idx 62 ulp=0.4935
     6.9143024614e-01f  // Idx 63 ulp=0.4996
};

// Intercept coefficients (b)
static const float coeffs_b[LUT_SIZE] = {
     8.4891917542e-01f,
     8.5265192470e-01f,
     8.5636530602e-01f,
     8.6005845496e-01f,
     8.6373049057e-01f,
     8.6738051511e-01f,
     8.7100761378e-01f,
     8.7461085450e-01f,
     8.7818928757e-01f,
     8.8174194547e-01f,
     8.8526784253e-01f,
     8.8876597467e-01f,
     8.9223531913e-01f,
     8.9567483416e-01f,
     8.9908345872e-01f,
     9.0246011223e-01f,
     9.0580369421e-01f,
     9.0911308403e-01f,
     9.1238714056e-01f,
     9.1562470186e-01f,
     9.1882458491e-01f,
     9.2198558523e-01f,
     9.2510647656e-01f,
     9.2818601059e-01f,
     9.3122291653e-01f,
     9.3421590085e-01f,
     9.3716364690e-01f,
     9.4006481455e-01f,
     9.4291803985e-01f,
     9.4572193468e-01f,
     9.4847508637e-01f,
     9.5117605733e-01f,
     9.5365976702e-01f,
     9.5625545218e-01f,
     9.5879458597e-01f,
     9.6127562900e-01f,
     9.6369701504e-01f,
     9.6605715067e-01f,
     9.6835441482e-01f,
     9.7058715837e-01f,
     9.7275370376e-01f,
     9.7485234454e-01f,
     9.7675557159e-01f,
     9.7859820623e-01f,
     9.8049184857e-01f,
     9.8231069294e-01f,
     9.8405288581e-01f,
     9.8571654213e-01f,
     9.8734796225e-01f,
     9.9006682244e-01f,
     9.9025990016e-01f,
     9.9158718058e-01f,
     9.9282594568e-01f,
     9.9393905844e-01f,
     9.9496634624e-01f,
     9.9593290801e-01f,
     9.9682465771e-01f,
     9.9758427254e-01f,
     9.9824179923e-01f,
     9.9879941088e-01f,
     9.9927025767e-01f,
     9.9961279336e-01f,
     9.9985650780e-01f,
     9.9999457184e-01f
};

/**
 * @brief Calculate LUT index from input value.
 * 
 * @param x Input value in [-0.99609375, -0.001953125]
 * @return int LUT index [0, 63]
 */
inline int get_lut_index(float x) {
    float normalized = (x - INTERVAL_START) / (INTERVAL_END - INTERVAL_START);
    int idx = static_cast<int>(normalized * LUT_SIZE);
    // Clamp to valid range
    if (idx < 0) idx = 0;
    if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;
    return idx;
}

/**
 * @brief Compute exp2(x) approximation using LUT.
 * 
 * @param x Input value in [-0.99609375, -0.001953125]
 * @return float Approximation of 2^x
 */
inline float exp2_approx(float x) {
    int idx = get_lut_index(x);
    return coeffs_a[idx] * x + coeffs_b[idx];
}

} // namespace bf16_exp2

#endif // BF16_EXP2_COEFFS_HPP
