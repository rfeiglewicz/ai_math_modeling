#ifndef BF16_EXP2_CORE_HPP
#define BF16_EXP2_CORE_HPP

#include "../utils/fp_utils.hpp"

// Tu w przyszłości dodasz nagłówki ac_types, np.:
// #include "ac_fixed.h"

/**
 * @brief Core approximation logic for exp2(x) in range [-9, 7].
 * 
 * This function will eventually use ac_datatypes for fixed-point arithmetic.
 * 
 * @param input_parts Decomposed BF16 input
 * @return FPRaw Decomposed BF16 result
 */
inline FPRaw bf16_exp2_core_approx(const FPRaw& input_parts) {
    FPRaw result = {};
    
    // Placeholder logic: currently returns 1.0
    // TODO: Implement LUT/Poly approximation using ac_types here
    
    result.sign = 0;        // Result is always positive
    result.exponent = 0;    // 2^0 = 1
    result.mantissa = 0;
    result.hidden_bit = 1;
    
    return result;
}

#endif // BF16_EXP2_CORE_HPP