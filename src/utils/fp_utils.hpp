#ifndef FP_UTILS_HPP
#define FP_UTILS_HPP

#include <cstdint>
#include <stdexcept>

// =========================================================
// Types and Constants
// =========================================================

enum class FPType {
    BF16,
    // Prepared for future expansion:
    // FP32,
    // FP16, 
    // FP8_E4M3, etc.
};

/**
 * @brief Configuration structure defining the geometry of a Floating Point format.
 * This allows the logic functions to be generic.
 */
struct FPConfig {
    uint32_t total_bits;
    uint32_t exp_bits;
    uint32_t mant_bits;
    int32_t  bias;

    // Helper: Returns a mask of 1s for the exponent size (e.g., 8 bits -> 0xFF)
    constexpr uint32_t exp_mask() const {
        return (1u << exp_bits) - 1u;
    }

    // Helper: Returns a mask of 1s for the mantissa size (e.g., 7 bits -> 0x7F)
    constexpr uint32_t mant_mask() const {
        return (1u << mant_bits) - 1u;
    }
};

/**
 * @brief Retrieval function to get parameters for a specific FPType.
 * To add a new format, simply add a case here.
 */
inline constexpr FPConfig get_fp_config(FPType type) {
    switch (type) {
        case FPType::BF16:
            // BF16: 16 bits total, 8 exp, 7 mant, bias 127
            return {16, 8, 7, 127};
        
        // Example for future FP32 support:
        // case FPType::FP32: return {32, 8, 23, 127};
        
        default:
            // Fallback/Error case (using BF16 layout to satisfy constexpr return)
             return {16, 8, 7, 127}; 
    }
}

struct FPStatus {
    bool is_zero;
    bool is_denormal;
    bool is_inf;
    bool is_nan;
};

struct FPRaw {
    bool sign;
    int32_t exponent;   // Unbiased exponent
    uint32_t mantissa;  // Explicit mantissa bits
    bool hidden_bit;    // The implicit 1 for normalized numbers

    FPStatus status;
};

// =========================================================
// Logic Functions (Generic Implementation)
// =========================================================

/**
 * @brief Classifies the raw bits based on the format configuration.
 */
inline FPStatus fp_classify(uint32_t raw_bits, FPType type) {
    FPStatus status = {false, false, false, false};
    
    // Get configuration for the requested type
    const FPConfig cfg = get_fp_config(type);

    // Filter valid bits for this format
    uint32_t payload_mask = (cfg.total_bits == 32) ? 0xFFFFFFFF : ((1u << cfg.total_bits) - 1);
    uint32_t masked_payload = raw_bits & payload_mask;

    // Extract raw fields using generic config
    uint32_t raw_mantissa = masked_payload & cfg.mant_mask();
    uint32_t raw_exp = (masked_payload >> cfg.mant_bits) & cfg.exp_mask();

    // Check classification logic (IEEE-754 standard style)
    if (raw_exp == 0) {
        if (raw_mantissa == 0) {
            status.is_zero = true;
        } else {
            status.is_denormal = true;
        }
    } else if (raw_exp == cfg.exp_mask()) { // Exponent is all 1s
        if (raw_mantissa == 0) {
            status.is_inf = true;
        } else {
            status.is_nan = true;
        }
    } 
    // Else: Normal number

    return status;
}

/**
 * @brief Decomposes payload into structural components using generic config.
 */
inline FPRaw fp_decompose(uint32_t payload, FPType type) {
    FPRaw result = {};
    const FPConfig cfg = get_fp_config(type);

    // 0. Classify
    result.status = fp_classify(payload, type);

    // 1. Extract Sign
    // Sign bit is always at position (total_bits - 1)
    result.sign = (payload >> (cfg.total_bits - 1)) & 0x1;

    // 2. Extract Raw Fields
    uint32_t raw_mantissa = payload & cfg.mant_mask();
    uint32_t raw_exp = (payload >> cfg.mant_bits) & cfg.exp_mask();

    result.mantissa = raw_mantissa;

    // 3. Logic for Unbiased Exponent and Hidden Bit
    if (result.status.is_zero || result.status.is_inf || result.status.is_nan) {
        result.exponent = 0; // Irrelevant for special cases
        result.hidden_bit = 0;
    } else if (result.status.is_denormal) {
        // Denormal: Exponent is fixed to (1 - Bias)
        result.exponent = 1 - cfg.bias;
        result.hidden_bit = 0;
    } else {
        // Normal: Unbiased = Raw - Bias
        result.exponent = static_cast<int32_t>(raw_exp) - cfg.bias;
        result.hidden_bit = 1;
    }

    return result;
}

/**
 * @brief Recomposes structure into raw bits using generic config.
 */
inline uint32_t fp_recompose(const FPRaw& components, FPType type) {
    const FPConfig cfg = get_fp_config(type);
    
    uint32_t payload = 0;
    uint32_t biased_exp = 0;
    uint32_t mantissa = 0;

    // A. Handle Special Cases (Priority)
    if (components.status.is_zero) {
        biased_exp = 0;
        mantissa = 0;
    } else if (components.status.is_inf) {
        biased_exp = cfg.exp_mask(); // All 1s
        mantissa = 0;
    } else if (components.status.is_nan) {
        biased_exp = cfg.exp_mask(); // All 1s
        // Maintain payload if present, else set default QNaN bit (MSB of mantissa)
        mantissa = (components.mantissa == 0) ? (1u << (cfg.mant_bits - 1)) : components.mantissa;
        mantissa &= cfg.mant_mask();
    } else {
        // B. Handle Normal/Denormal
        if (components.status.is_denormal) {
            biased_exp = 0;
            mantissa = components.mantissa;
        } else {
            // Normal
            int32_t temp_exp = components.exponent + cfg.bias;

            // Simple saturation logic for modeling purposes
            if (temp_exp <= 0) {
                // Underflow to zero (simplified)
                biased_exp = 0; 
                mantissa = 0; 
            } else if (temp_exp >= (int32_t)cfg.exp_mask()) {
                // Overflow to Infinity
                biased_exp = cfg.exp_mask();
                mantissa = 0;
            } else {
                biased_exp = static_cast<uint32_t>(temp_exp);
                mantissa = components.mantissa;
            }
        }
    }

    // 1. Pack Sign
    if (components.sign) {
        payload |= (1u << (cfg.total_bits - 1));
    }

    // 2. Pack Exponent
    payload |= ((biased_exp & cfg.exp_mask()) << cfg.mant_bits);

    // 3. Pack Mantissa
    payload |= (mantissa & cfg.mant_mask());

    // Final Mask to ensure cleanliness
    uint32_t total_mask = (cfg.total_bits == 32) ? 0xFFFFFFFF : ((1u << cfg.total_bits) - 1);
    return payload & total_mask;
}

#endif // FP_UTILS_HPP