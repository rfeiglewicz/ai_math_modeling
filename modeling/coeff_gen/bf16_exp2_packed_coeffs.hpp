#ifndef BF16_EXP2_PACKED_COEFFS_HPP
#define BF16_EXP2_PACKED_COEFFS_HPP

#include "ac_int.h"
#include "ac_fixed.h"

namespace bf16_exp2_packed {

constexpr int LUT_SIZE = 128;
constexpr int COEFF_I = 1;
constexpr int COEFF_F = 20;
constexpr int COEFF_W = 21;
constexpr int PACKED_W = 42;

constexpr int LOG2E_I = 1;
constexpr int LOG2E_F = 22;
constexpr int LOG2E_W = 23;

// Log2(e) in 1.25 format
// Value: 1.44269
static const ac_int<LOG2E_W, false> log2e_int_val = 0x5c551d;

// Packed coefficients: [ b (26 bits) | a (26 bits) ]
// Format: unsigned 1.14
static const ac_int<PACKED_W, false> coeffs[LUT_SIZE] = {
    0x1b22a659157ULL, // Index 0
    0x1b31fa59915ULL, // Index 1
    0x1b41445a0ddULL, // Index 2
    0x1b50865a8b0ULL, // Index 3
    0x1b5fba5b08dULL, // Index 4
    0x1b6ee65b876ULL, // Index 5
    0x1b7e065c06aULL, // Index 6
    0x1b8d1c5c868ULL, // Index 7
    0x1b9c265d072ULL, // Index 8
    0x1bab265d887ULL, // Index 9
    0x1bba185e0a8ULL, // Index 10
    0x1bc9005e8d3ULL, // Index 11
    0x1bd7dc5f10aULL, // Index 12
    0x1be6aa5f94cULL, // Index 13
    0x1bf56c6019aULL, // Index 14
    0x1c0422609f4ULL, // Index 15
    0x1c12ca61258ULL, // Index 16
    0x1c216661ac9ULL, // Index 17
    0x1c2ff262345ULL, // Index 18
    0x1c3e7262bceULL, // Index 19
    0x1c4ce463462ULL, // Index 20
    0x1c5b4663d02ULL, // Index 21
    0x1c699a645aeULL, // Index 22
    0x1c77e064e66ULL, // Index 23
    0x1c86166572aULL, // Index 24
    0x1c943c65ffaULL, // Index 25
    0x1ca252668d6ULL, // Index 26
    0x1cb058671bfULL, // Index 27
    0x1cbe4e67ab5ULL, // Index 28
    0x1ccc34683b6ULL, // Index 29
    0x1cda0a68cc4ULL, // Index 30
    0x1ce7cc695dfULL, // Index 31
    0x1cf57e69f07ULL, // Index 32
    0x1d031e6a83bULL, // Index 33
    0x1d10ae6b17cULL, // Index 34
    0x1d1e286bacaULL, // Index 35
    0x1d2b926c424ULL, // Index 36
    0x1d38e86cd8cULL, // Index 37
    0x1d462c6d701ULL, // Index 38
    0x1d535a6e083ULL, // Index 39
    0x1d60766ea12ULL, // Index 40
    0x1d6d7c6f3afULL, // Index 41
    0x1d7a706fd59ULL, // Index 42
    0x1d874e70710ULL, // Index 43
    0x1d9416710d5ULL, // Index 44
    0x1da0c871aa7ULL, // Index 45
    0x1dad6672487ULL, // Index 46
    0x1db9ec72e75ULL, // Index 47
    0x1dc65e73870ULL, // Index 48
    0x1dd2b67427aULL, // Index 49
    0x1ddefa74c91ULL, // Index 50
    0x1deb24756b7ULL, // Index 51
    0x1df736760eaULL, // Index 52
    0x1e033276b2cULL, // Index 53
    0x1e0f147757cULL, // Index 54
    0x1e1adc77fdaULL, // Index 55
    0x1e268c78a47ULL, // Index 56
    0x1e3222794c2ULL, // Index 57
    0x1e3da079f4cULL, // Index 58
    0x1e49007a9e4ULL, // Index 59
    0x1e54487b48bULL, // Index 60
    0x1e5f747bf41ULL, // Index 61
    0x1e6a847ca06ULL, // Index 62
    0x1e75787d4daULL, // Index 63
    0x1e7ef67de60ULL, // Index 64
    0x1e89b67e950ULL, // Index 65
    0x1e94587f44fULL, // Index 66
    0x1e9edc7ff5eULL, // Index 67
    0x1ea94280a7cULL, // Index 68
    0x1eb38a815a9ULL, // Index 69
    0x1ebdb4820e6ULL, // Index 70
    0x1ec7be82c33ULL, // Index 71
    0x1ed1aa8378fULL, // Index 72
    0x1edb76842fbULL, // Index 73
    0x1ee52084e77ULL, // Index 74
    0x1eeeaa85a03ULL, // Index 75
    0x1ef8128659fULL, // Index 76
    0x1f015a8714bULL, // Index 77
    0x1f0a8087d07ULL, // Index 78
    0x1f1382888d4ULL, // Index 79
    0x1f1c62894b1ULL, // Index 80
    0x1f251e8a09eULL, // Index 81
    0x1f2db68ac9cULL, // Index 82
    0x1f362c8b8abULL, // Index 83
    0x1f3e7a8c4cbULL, // Index 84
    0x1f45a28cf74ULL, // Index 85
    0x1f4cac8da2bULL, // Index 86
    0x1f54948e679ULL, // Index 87
    0x1f5c568f2d8ULL, // Index 88
    0x1f63f08ff48ULL, // Index 89
    0x1f6b6490bcaULL, // Index 90
    0x1f72b09185dULL, // Index 91
    0x1f79d292502ULL, // Index 92
    0x1f80cc931b8ULL, // Index 93
    0x1f879c93e80ULL, // Index 94
    0x1f8e4494b59ULL, // Index 95
    0x1f952695914ULL, // Index 96
    0x1f9b7496612ULL, // Index 97
    0x1fa19897323ULL, // Index 98
    0x1fa78e98046ULL, // Index 99
    0x1fad5a98d7bULL, // Index 100
    0x1fb2f699ac2ULL, // Index 101
    0x1fb8669a81cULL, // Index 102
    0x1fbda89b588ULL, // Index 103
    0x1fc2ba9c307ULL, // Index 104
    0x1fc79e9d099ULL, // Index 105
    0x1fcc089dd63ULL, // Index 106
    0x1fd0489ea3dULL, // Index 107
    0x1fd4a29f806ULL, // Index 108
    0x1fd8c8a05e1ULL, // Index 109
    0x1fdcbea13d0ULL, // Index 110
    0x1fe082a21d2ULL, // Index 111
    0x1fe43aa3090ULL, // Index 112
    0x1fe788a3e82ULL, // Index 113
    0x1feaaea4cc0ULL, // Index 114
    0x1feda0a5b11ULL, // Index 115
    0x1ff05ca6976ULL, // Index 116
    0x1ff2cea777bULL, // Index 117
    0x1ff50ea8593ULL, // Index 118
    0x1ff728a9433ULL, // Index 119
    0x1ff91eaa37aULL, // Index 120
    0x1ffabcab1ecULL, // Index 121
    0x1ffc28ac08eULL, // Index 122
    0x1ffd5cacf44ULL, // Index 123
    0x1ffe62aded1ULL, // Index 124
    0x1fff1eaed67ULL, // Index 125
    0x1fffa8afce8ULL, // Index 126
    0x1ffff2b0c81ULL // Index 127
};

} // namespace bf16_exp2_packed

#endif // BF16_EXP2_PACKED_COEFFS_HPP
