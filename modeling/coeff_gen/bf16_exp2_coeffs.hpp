#ifndef BF16_EXP2_COEFFS_HPP
#define BF16_EXP2_COEFFS_HPP

#include <cstdint>

// =============================================================
// exp2(x) Linear Approximation Coefficients for BF16
// =============================================================
//
// Generated: 2025-12-21 16:39:01
// Interval:  [-0.2490234375, -0.001953125]
// LUT Size:  16
//
// Approximation: y = a * x + b
//
// Error Metrics (BF16 ULP):
//   Worst-case: 0.499632 ULP
//   Average:    0.266672 ULP
// =============================================================

namespace bf16_exp2 {

constexpr int LUT_SIZE = 16;
constexpr float INTERVAL_START = -0.2490234375f;
constexpr float INTERVAL_END = -0.001953125f;

// Slope coefficients (a)
static const float coeffs_a[LUT_SIZE] = {
     5.9730334045e-01f,  // [-0.249023, -0.195312] ulp=0.4992
     6.1719785143e-01f,  // [-0.194336, -0.140625] ulp=0.4984
     6.3703975767e-01f,  // [-0.139648, -0.105469] ulp=0.4925
     6.5759421338e-01f,  // [-0.104980, -0.078125] ulp=0.4943
     6.6184174413e-01f,  // [-0.077637, -0.056641] ulp=0.4926
     6.6968485487e-01f,  // [-0.056396, -0.042969] ulp=0.4878
     6.7595242783e-01f,  // [-0.042725, -0.030273] ulp=0.4991
     6.8039290527e-01f,  // [-0.030151, -0.023438] ulp=0.4914
     6.8362446835e-01f,  // [-0.023315, -0.016602] ulp=0.4935
     6.8622317019e-01f,  // [-0.016479, -0.012695] ulp=0.4996
     6.8790346076e-01f,  // [-0.012634, -0.009277] ulp=0.3591
     6.8932304260e-01f,  // [-0.009216, -0.006836] ulp=0.4990
     6.9028665137e-01f,  // [-0.006805, -0.005127] ulp=0.2047
     6.9106331392e-01f,  // [-0.005096, -0.003662] ulp=0.3510
     6.9159838531e-01f,  // [-0.003647, -0.002808] ulp=0.4996
     6.9200813266e-01f  // [-0.002792, -0.001953] ulp=0.4950
};

// Intercept coefficients (b)
static const float coeffs_b[LUT_SIZE] = {
     9.8995698768e-01f,
     9.9381819585e-01f,
     9.9665230223e-01f,
     9.9879771967e-01f,
     9.9897397779e-01f,
     9.9942416193e-01f,
     9.9969317419e-01f,
     9.9983058023e-01f,
     9.9990611577e-01f,
     9.9995019818e-01f,
     9.9997154431e-01f,
     9.9998487287e-01f,
     9.9999153094e-01f,
     9.9999552413e-01f,
     9.9999751636e-01f,
     9.9999866363e-01f
};

/**
 * @brief Calculate LUT index from input value.
 * 
 * @param x Input value in [-0.2490234375, -0.001953125]
 * @return int LUT index [0, 15]
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
 * @param x Input value in [-0.2490234375, -0.001953125]
 * @return float Approximation of 2^x
 */
inline float exp2_approx(float x) {
    int idx = get_lut_index(x);
    return coeffs_a[idx] * x + coeffs_b[idx];
}

} // namespace bf16_exp2

#endif // BF16_EXP2_COEFFS_HPP
