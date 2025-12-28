#ifndef BF16_EXP2_PACKED_COEFFS_HPP
#define BF16_EXP2_PACKED_COEFFS_HPP

#include "ac_int.h"
#include "ac_fixed.h"

namespace bf16_exp2_packed {

constexpr int LUT_SIZE = 128;
constexpr int COEFF_I = 1;
constexpr int COEFF_F = 25;
constexpr int COEFF_W = 26;
constexpr int PACKED_W = 52;

constexpr int LOG2E_I = 1;
constexpr int LOG2E_F = 22;
constexpr int LOG2E_W = 23;

// Log2(e) in 1.25 format
// Value: 1.44269
static const ac_int<LOG2E_W, false> log2e_int_val = 0x5c551d;

// Packed coefficients: [ b (26 bits) | a (26 bits) ]
// Format: unsigned 1.19
static const ac_int<PACKED_W, false> coeffs[LUT_SIZE] = {
    0x6c8a9f8b22afaULL, // Index 0
    0x6cc7ef8b322a4ULL, // Index 1
    0x6d05178b41ba6ULL, // Index 2
    0x6d42180b51602ULL, // Index 3
    0x6d7eef0b611baULL, // Index 4
    0x6dbb9c0b70ed0ULL, // Index 5
    0x6df81e0b80d46ULL, // Index 6
    0x6e34748b90d1eULL, // Index 7
    0x6e709e8ba0e59ULL, // Index 8
    0x6eac9a8bb10faULL, // Index 9
    0x6ee8678bc1502ULL, // Index 10
    0x6f24058bd1a74ULL, // Index 11
    0x6f5f728be2151ULL, // Index 12
    0x6f9aae8bf299bULL, // Index 13
    0x6fd5b78c03355ULL, // Index 14
    0x70108d0c13e81ULL, // Index 15
    0x704b2e0c24b1fULL, // Index 16
    0x7085998c35934ULL, // Index 17
    0x70bfce8c468bfULL, // Index 18
    0x70f9cc0c579c4ULL, // Index 19
    0x7133910c68c45ULL, // Index 20
    0x716d1c0c7a043ULL, // Index 21
    0x71a66c0c8b5c0ULL, // Index 22
    0x71df808c9ccc0ULL, // Index 23
    0x7218580cae543ULL, // Index 24
    0x7250f20cbff4cULL, // Index 25
    0x72894c0cd1addULL, // Index 26
    0x72c1668ce37f9ULL, // Index 27
    0x72f93f8cf56a1ULL, // Index 28
    0x7330d58d076d7ULL, // Index 29
    0x7368288d1989eULL, // Index 30
    0x739f368d2bbf7ULL, // Index 31
    0x73d5fe8d3e0e6ULL, // Index 32
    0x740c7f8d5076cULL, // Index 33
    0x7442b80d62f8bULL, // Index 34
    0x7478a78d75946ULL, // Index 35
    0x74ae4c0d8849eULL, // Index 36
    0x74e3a48d9b197ULL, // Index 37
    0x7518b00dae032ULL, // Index 38
    0x754d6d0dc1072ULL, // Index 39
    0x7581da0dd4259ULL, // Index 40
    0x75b5f68de75e9ULL, // Index 41
    0x75e9c10dfab24ULL, // Index 42
    0x761d380e0e20eULL, // Index 43
    0x76505a0e21aa7ULL, // Index 44
    0x7683260e354f3ULL, // Index 45
    0x76b59a8e490f4ULL, // Index 46
    0x76e7b68e5ceadULL, // Index 47
    0x7719780e70e1eULL, // Index 48
    0x774ade8e84f4cULL, // Index 49
    0x777be80e99239ULL, // Index 50
    0x77ac938ead6e6ULL, // Index 51
    0x77dcdf8ec1d56ULL, // Index 52
    0x780cca0ed658dULL, // Index 53
    0x783c528eeaf8bULL, // Index 54
    0x786b778effb55ULL, // Index 55
    0x789a370f148ebULL, // Index 56
    0x78c88f8f29851ULL, // Index 57
    0x78f6800f3e98aULL, // Index 58
    0x7924070f53c97ULL, // Index 59
    0x7951228f6917bULL, // Index 60
    0x797dd18f7e83aULL, // Index 61
    0x79aa128f940d4ULL, // Index 62
    0x79d5e38fa9b4eULL, // Index 63
    0x79fbdf8fbcc09ULL, // Index 64
    0x7a26db0fd2a0cULL, // Index 65
    0x7a51620fe89f5ULL, // Index 66
    0x7a7b738ffebc8ULL, // Index 67
    0x7aa50d9014f86ULL, // Index 68
    0x7ace2e902b534ULL, // Index 69
    0x7af6d51041cd2ULL, // Index 70
    0x7b1eff1058664ULL, // Index 71
    0x7b46ab106f1ecULL, // Index 72
    0x7b6dd81085f6eULL, // Index 73
    0x7b9483109ceecULL, // Index 74
    0x7bbaab90b406aULL, // Index 75
    0x7be04f10cb3e8ULL, // Index 76
    0x7c056c10e296cULL, // Index 77
    0x7c2a0110fa0f6ULL, // Index 78
    0x7c4e0c1111a8aULL, // Index 79
    0x7c718b912962aULL, // Index 80
    0x7c947d11413dcULL, // Index 81
    0x7cb6df915939eULL, // Index 82
    0x7cd8b11171578ULL, // Index 83
    0x7cf9ef9189968ULL, // Index 84
    0x7d1688119ee90ULL, // Index 85
    0x7d32b591b4562ULL, // Index 86
    0x7d525611ccf24ULL, // Index 87
    0x7d715c91e5b0cULL, // Index 88
    0x7d8fc791fe918ULL, // Index 89
    0x7dad94921794eULL, // Index 90
    0x7dcac19230bb0ULL, // Index 91
    0x7de74d124a042ULL, // Index 92
    0x7e03351263706ULL, // Index 93
    0x7e1e77127d000ULL, // Index 94
    0x7e39119296b32ULL, // Index 95
    0x7e549c12b228cULL, // Index 96
    0x7e6dd612cc25cULL, // Index 97
    0x7e866212e646eULL, // Index 98
    0x7e9e3e13008c6ULL, // Index 99
    0x7eb568131af66ULL, // Index 100
    0x7ecbdd9335850ULL, // Index 101
    0x7ee19c935038aULL, // Index 102
    0x7ef6a3936b116ULL, // Index 103
    0x7f0aef13860f8ULL, // Index 104
    0x7f1e7e93a1330ULL, // Index 105
    0x7f302493bac66ULL, // Index 106
    0x7f412593d47baULL, // Index 107
    0x7f528993f00c2ULL, // Index 108
    0x7f6327940bc32ULL, // Index 109
    0x7f72fd9427a08ULL, // Index 110
    0x7f820a1443a4aULL, // Index 111
    0x7f90ee1461216ULL, // Index 112
    0x7f9e23147d054ULL, // Index 113
    0x7faabc1499802ULL, // Index 114
    0x7fb68214b622aULL, // Index 115
    0x7fc17194d2eceULL, // Index 116
    0x7fcb3a94eef6eULL, // Index 117
    0x7fd438150b26cULL, // Index 118
    0x7fdca59528674ULL, // Index 119
    0x7fe4789546f44ULL, // Index 120
    0x7feaf79563d8eULL, // Index 121
    0x7ff0a515811d6ULL, // Index 122
    0x7ff574159e89eULL, // Index 123
    0x7ff98b95bda28ULL, // Index 124
    0x7ffc7b95dace4ULL, // Index 125
    0x7ffea215f9d0eULL, // Index 126
    0x7fffcf1619026ULL // Index 127
};

} // namespace bf16_exp2_packed

#endif // BF16_EXP2_PACKED_COEFFS_HPP
