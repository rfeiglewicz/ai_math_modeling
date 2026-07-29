// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_exp2.h for the primary calling header

#include "Vbf16_exp2__pch.h"
#include "Vbf16_exp2___024root.h"

VL_ATTR_COLD void Vbf16_exp2___024root___eval_static(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vbf16_exp2___024root___eval_initial__TOP(Vbf16_exp2___024root* vlSelf);

VL_ATTR_COLD void Vbf16_exp2___024root___eval_initial(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_initial\n"); );
    // Body
    Vbf16_exp2___024root___eval_initial__TOP(vlSelf);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = vlSelf->rst_n;
}

VL_ATTR_COLD void Vbf16_exp2___024root___eval_initial__TOP(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_initial__TOP\n"); );
    // Body
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0U] = 0x1b22a659157ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[1U] = 0x1b31fa59915ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[2U] = 0x1b41445a0ddULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[3U] = 0x1b50865a8b0ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[4U] = 0x1b5fba5b08dULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[5U] = 0x1b6ee65b876ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[6U] = 0x1b7e065c06aULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[7U] = 0x1b8d1c5c868ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[8U] = 0x1b9c265d072ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[9U] = 0x1bab265d887ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xaU] = 0x1bba185e0a8ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xbU] = 0x1bc9005e8d3ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xcU] = 0x1bd7dc5f10aULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xdU] = 0x1be6aa5f94cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xeU] = 0x1bf56c6019aULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xfU] = 0x1c0422609f4ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x10U] = 0x1c12ca61258ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x11U] = 0x1c216661ac9ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x12U] = 0x1c2ff262345ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x13U] = 0x1c3e7262bceULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x14U] = 0x1c4ce463462ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x15U] = 0x1c5b4663d02ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x16U] = 0x1c699a645aeULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x17U] = 0x1c77e064e66ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x18U] = 0x1c86166572aULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x19U] = 0x1c943c65ffaULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1aU] = 0x1ca252668d6ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1bU] = 0x1cb058671bfULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1cU] = 0x1cbe4e67ab5ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1dU] = 0x1ccc34683b6ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1eU] = 0x1cda0a68cc4ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1fU] = 0x1ce7cc695dfULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x20U] = 0x1cf57e69f07ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x21U] = 0x1d031e6a83bULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x22U] = 0x1d10ae6b17cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x23U] = 0x1d1e286bacaULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x24U] = 0x1d2b926c424ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x25U] = 0x1d38e86cd8cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x26U] = 0x1d462c6d701ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x27U] = 0x1d535a6e083ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x28U] = 0x1d60766ea12ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x29U] = 0x1d6d7c6f3afULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2aU] = 0x1d7a706fd59ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2bU] = 0x1d874e70710ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2cU] = 0x1d9416710d5ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2dU] = 0x1da0c871aa7ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2eU] = 0x1dad6672487ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2fU] = 0x1db9ec72e75ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x30U] = 0x1dc65e73870ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x31U] = 0x1dd2b67427aULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x32U] = 0x1ddefa74c91ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x33U] = 0x1deb24756b7ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x34U] = 0x1df736760eaULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x35U] = 0x1e033276b2cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x36U] = 0x1e0f147757cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x37U] = 0x1e1adc77fdaULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x38U] = 0x1e268c78a47ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x39U] = 0x1e3222794c2ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3aU] = 0x1e3da079f4cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3bU] = 0x1e49007a9e4ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3cU] = 0x1e54487b48bULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3dU] = 0x1e5f747bf41ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3eU] = 0x1e6a847ca06ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3fU] = 0x1e75787d4daULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x40U] = 0x1e7ef67de60ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x41U] = 0x1e89b67e950ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x42U] = 0x1e94587f44fULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x43U] = 0x1e9edc7ff5eULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x44U] = 0x1ea94280a7cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x45U] = 0x1eb38a815a9ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x46U] = 0x1ebdb4820e6ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x47U] = 0x1ec7be82c33ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x48U] = 0x1ed1aa8378fULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x49U] = 0x1edb76842fbULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4aU] = 0x1ee52084e77ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4bU] = 0x1eeeaa85a03ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4cU] = 0x1ef8128659fULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4dU] = 0x1f015a8714bULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4eU] = 0x1f0a8087d07ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4fU] = 0x1f1382888d4ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x50U] = 0x1f1c62894b1ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x51U] = 0x1f251e8a09eULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x52U] = 0x1f2db68ac9cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x53U] = 0x1f362c8b8abULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x54U] = 0x1f3e7a8c4cbULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x55U] = 0x1f45a28cf74ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x56U] = 0x1f4cac8da2bULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x57U] = 0x1f54948e679ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x58U] = 0x1f5c568f2d8ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x59U] = 0x1f63f08ff48ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5aU] = 0x1f6b6490bcaULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5bU] = 0x1f72b09185dULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5cU] = 0x1f79d292502ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5dU] = 0x1f80cc931b8ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5eU] = 0x1f879c93e80ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5fU] = 0x1f8e4494b59ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x60U] = 0x1f952695914ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x61U] = 0x1f9b7496612ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x62U] = 0x1fa19897323ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x63U] = 0x1fa78e98046ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x64U] = 0x1fad5a98d7bULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x65U] = 0x1fb2f699ac2ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x66U] = 0x1fb8669a81cULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x67U] = 0x1fbda89b588ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x68U] = 0x1fc2ba9c307ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x69U] = 0x1fc79e9d099ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6aU] = 0x1fcc089dd63ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6bU] = 0x1fd0489ea3dULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6cU] = 0x1fd4a29f806ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6dU] = 0x1fd8c8a05e1ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6eU] = 0x1fdcbea13d0ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6fU] = 0x1fe082a21d2ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x70U] = 0x1fe43aa3090ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x71U] = 0x1fe788a3e82ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x72U] = 0x1feaaea4cc0ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x73U] = 0x1feda0a5b11ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x74U] = 0x1ff05ca6976ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x75U] = 0x1ff2cea777bULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x76U] = 0x1ff50ea8593ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x77U] = 0x1ff728a9433ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x78U] = 0x1ff91eaa37aULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x79U] = 0x1ffabcab1ecULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7aU] = 0x1ffc28ac08eULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7bU] = 0x1ffd5cacf44ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7cU] = 0x1ffe62aded1ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7dU] = 0x1fff1eaed67ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7eU] = 0x1fffa8afce8ULL;
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7fU] = 0x1ffff2b0c81ULL;
}

VL_ATTR_COLD void Vbf16_exp2___024root___eval_final(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_exp2___024root___dump_triggers__stl(Vbf16_exp2___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vbf16_exp2___024root___eval_phase__stl(Vbf16_exp2___024root* vlSelf);

VL_ATTR_COLD void Vbf16_exp2___024root___eval_settle(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_settle\n"); );
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelf->__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY((0x64U < __VstlIterCount))) {
#ifdef VL_DEBUG
            Vbf16_exp2___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_exp2.sv", 30, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vbf16_exp2___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_exp2___024root___dump_triggers__stl(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vbf16_exp2___024root___stl_sequent__TOP__0(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___stl_sequent__TOP__0\n"); );
    // Init
    QData/*58:0*/ bf16_exp2__DOT__u_lin_approx__DOT__ax_unsigned;
    bf16_exp2__DOT__u_lin_approx__DOT__ax_unsigned = 0;
    QData/*61:0*/ bf16_exp2__DOT__u_lin_approx__DOT__neg_ax;
    bf16_exp2__DOT__u_lin_approx__DOT__neg_ax = 0;
    QData/*61:0*/ bf16_exp2__DOT__u_lin_approx__DOT__b_aligned;
    bf16_exp2__DOT__u_lin_approx__DOT__b_aligned = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_normalize__DOT__shift_amt;
    bf16_exp2__DOT__u_normalize__DOT__shift_amt = 0;
    QData/*61:0*/ bf16_exp2__DOT__u_normalize__DOT__shifted_res;
    bf16_exp2__DOT__u_normalize__DOT__shifted_res = 0;
    IData/*31:0*/ bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i;
    bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0;
    CData/*0:0*/ bf16_exp2__DOT__u_round__DOT__lsb_bit;
    bf16_exp2__DOT__u_round__DOT__lsb_bit = 0;
    CData/*0:0*/ bf16_exp2__DOT__u_round__DOT__guard_bit;
    bf16_exp2__DOT__u_round__DOT__guard_bit = 0;
    CData/*0:0*/ bf16_exp2__DOT__u_round__DOT__round_up;
    bf16_exp2__DOT__u_round__DOT__round_up = 0;
    QData/*58:0*/ bf16_exp2__DOT__u_round__DOT__sticky_masked;
    bf16_exp2__DOT__u_round__DOT__sticky_masked = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_round__DOT__result_m_ext;
    bf16_exp2__DOT__u_round__DOT__result_m_ext = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_round__DOT__adjusted_exp;
    bf16_exp2__DOT__u_round__DOT__adjusted_exp = 0;
    CData/*7:0*/ bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0;
    CData/*6:0*/ bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0;
    // Body
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
        = vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg
        [0U];
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
        = (0x7ffffffffffffffULL & (vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
                                   + VL_SHIFTL_QQI(59,59,32, 
                                                   vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg
                                                   [1U], 0xfU)));
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
        = (0x7ffffffffffffffULL & (vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
                                   + VL_SHIFTL_QQI(59,59,32, 
                                                   vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg
                                                   [2U], 0x1eU)));
    vlSelf->m_axis_tdata = vlSelf->bf16_exp2__DOT__bf16_out;
    vlSelf->bf16_exp2__DOT__exp_delay[0U] = (0x1ffU 
                                             & (vlSelf->bf16_exp2__DOT__s1_decomposed 
                                                >> 0xcU));
    vlSelf->bf16_exp2__DOT__s3_mant_src = ((0x80U & 
                                            (vlSelf->bf16_exp2__DOT__s1_decomposed 
                                             << 3U)) 
                                           | (0x7fU 
                                              & (vlSelf->bf16_exp2__DOT__s1_decomposed 
                                                 >> 5U)));
    vlSelf->bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb = 0x1ffU;
    bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0x3dU;
    {
        while (VL_LTES_III(32, 0U, bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) {
            if (((0x3dU >= (0x3fU & bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) 
                 && (1U & (IData)((vlSelf->bf16_exp2__DOT__s5_unnorm_res 
                                   >> (0x3fU & bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)))))) {
                vlSelf->bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb 
                    = (0x1ffU & bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i);
                goto __Vlabel1;
            }
            bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                = (bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                   - (IData)(1U));
        }
        __Vlabel1: ;
    }
    vlSelf->bf16_exp2__DOT__u_decompose__DOT__status_comb 
        = (((IData)(((0x7f80U == (0x7f80U & (IData)(vlSelf->s_axis_tdata))) 
                     & (0U != (0x7fU & (IData)(vlSelf->s_axis_tdata))))) 
            << 3U) | (((IData)((0x7f80U == (0x7fffU 
                                            & (IData)(vlSelf->s_axis_tdata)))) 
                       << 2U) | (((IData)(((0U == (0x7f80U 
                                                   & (IData)(vlSelf->s_axis_tdata))) 
                                           & (0U != 
                                              (0x7fU 
                                               & (IData)(vlSelf->s_axis_tdata))))) 
                                  << 1U) | (IData)(
                                                   (0U 
                                                    == 
                                                    (0x7fffU 
                                                     & (IData)(vlSelf->s_axis_tdata)))))));
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[0U] 
        = (0xfffffffffULL & ((QData)((IData)((0x1fffffU 
                                              & (IData)(vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff)))) 
                             * (QData)((IData)((0x7fffU 
                                                & (IData)(vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__frac_aligned))))));
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[1U] 
        = (0xfffffffffULL & ((QData)((IData)((0x1fffffU 
                                              & (IData)(vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff)))) 
                             * (QData)((IData)((0x7fffU 
                                                & (IData)(
                                                          (vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__frac_aligned 
                                                           >> 0xfU)))))));
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[2U] 
        = (0xfffffffffULL & ((QData)((IData)((0x1fffffU 
                                              & (IData)(vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff)))) 
                             * (QData)((IData)((0xffU 
                                                & (IData)(
                                                          (vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__frac_aligned 
                                                           >> 0x1eU)))))));
    vlSelf->m_axis_tvalid = (1U & ((IData)(vlSelf->bf16_exp2__DOT__gen_axi_pipelined__DOT__vld_sr) 
                                   >> 0xfU));
    bf16_exp2__DOT__u_normalize__DOT__shift_amt = (0x1ffU 
                                                   & ((IData)(0x3dU) 
                                                      - (IData)(vlSelf->bf16_exp2__DOT__u_normalize__DOT__msb_idx_s)));
    bf16_exp2__DOT__u_normalize__DOT__shifted_res = 
        (0x3fffffffffffffffULL & VL_SHIFTL_QQI(62,62,9, vlSelf->bf16_exp2__DOT__u_normalize__DOT__res_s, (IData)(bf16_exp2__DOT__u_normalize__DOT__shift_amt)));
    vlSelf->bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb 
        = (0x7ffffffffffffffULL & (bf16_exp2__DOT__u_normalize__DOT__shifted_res 
                                   >> 3U));
    bf16_exp2__DOT__u_round__DOT__result_m_ext = vlSelf->bf16_exp2__DOT__u_round__DOT__p2_m_ext;
    bf16_exp2__DOT__u_round__DOT__adjusted_exp = vlSelf->bf16_exp2__DOT__u_round__DOT__p2_base_exp;
    if ((0x100U & (IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext))) {
        bf16_exp2__DOT__u_round__DOT__adjusted_exp 
            = (0x1ffU & ((IData)(1U) + (IData)(bf16_exp2__DOT__u_round__DOT__adjusted_exp)));
        bf16_exp2__DOT__u_round__DOT__result_m_ext 
            = (0x1ffU & VL_SHIFTR_III(9,9,32, (IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext), 1U));
    }
    vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb = 0U;
    vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb 
        = (0x1fffffU & vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb);
    if ((0U == (IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext))) {
        vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb 
            = (1U | vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb);
    } else if (((IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p2_is_sub) 
                & (~ ((IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext) 
                      >> 7U)))) {
        vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb 
            = ((0x20000fU & vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb) 
               | (0x181000U | (0xfe0U & ((IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext) 
                                         << 5U))));
        vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb 
            = (2U | vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb);
    } else {
        vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb 
            = ((0x20000fU & vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb) 
               | (0x10U | (((IData)(bf16_exp2__DOT__u_round__DOT__adjusted_exp) 
                            << 0xcU) | (0xfe0U & ((IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext) 
                                                  << 5U)))));
    }
    bf16_exp2__DOT__u_lin_approx__DOT__ax_unsigned 
        = vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__merged_q;
    bf16_exp2__DOT__u_lin_approx__DOT__neg_ax = (0x3fffffffffffffffULL 
                                                 & (- 
                                                    VL_EXTENDS_QQ(62,60, bf16_exp2__DOT__u_lin_approx__DOT__ax_unsigned)));
    bf16_exp2__DOT__u_lin_approx__DOT__b_aligned = 
        (0x3fffffffffffffffULL & ((QData)((IData)(vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__coeff_b_stage)) 
                                  << 0x26U));
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__calc_res 
        = (0x3fffffffffffffffULL & (bf16_exp2__DOT__u_lin_approx__DOT__b_aligned 
                                    + bf16_exp2__DOT__u_lin_approx__DOT__neg_ax));
    vlSelf->bf16_exp2__DOT__int_delay[0U] = vlSelf->bf16_exp2__DOT__s4_int_part;
    bf16_exp2__DOT__u_round__DOT__sticky_masked = (vlSelf->bf16_exp2__DOT__u_round__DOT__p1_mant 
                                                   & (((QData)((IData)(
                                                                       VL_LTS_III(9, 0x3bU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                       << 0x3aU) 
                                                      | (((QData)((IData)(
                                                                          VL_LTS_III(9, 0x3aU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                          << 0x39U) 
                                                         | (((QData)((IData)(
                                                                             VL_LTS_III(9, 0x39U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                             << 0x38U) 
                                                            | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x38U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                << 0x37U) 
                                                               | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x37U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                   << 0x36U) 
                                                                  | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x36U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                      << 0x35U) 
                                                                     | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x35U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                         << 0x34U) 
                                                                        | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x34U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                            << 0x33U) 
                                                                           | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x33U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                               << 0x32U) 
                                                                              | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x32U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x31U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x31U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x30U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x30U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2fU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2fU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2eU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2eU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2dU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2dU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2cU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2cU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2bU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2bU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2aU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2aU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x29U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x29U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x28U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x28U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x27U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x27U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x26U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x26U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x25U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x25U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x24U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x24U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x23U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x23U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x22U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x22U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x21U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x21U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x20U) 
                                                                                | (QData)((IData)(
                                                                                ((VL_LTS_III(9, 0x20U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1fU) 
                                                                                | ((VL_LTS_III(9, 0x1fU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1eU) 
                                                                                | ((VL_LTS_III(9, 0x1eU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1dU) 
                                                                                | ((VL_LTS_III(9, 0x1dU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1cU) 
                                                                                | ((VL_LTS_III(9, 0x1cU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1bU) 
                                                                                | ((VL_LTS_III(9, 0x1bU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1aU) 
                                                                                | ((VL_LTS_III(9, 0x1aU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x19U) 
                                                                                | ((VL_LTS_III(9, 0x19U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x18U) 
                                                                                | ((VL_LTS_III(9, 0x18U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x17U) 
                                                                                | ((VL_LTS_III(9, 0x17U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x16U) 
                                                                                | ((VL_LTS_III(9, 0x16U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x15U) 
                                                                                | ((VL_LTS_III(9, 0x15U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x14U) 
                                                                                | ((VL_LTS_III(9, 0x14U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x13U) 
                                                                                | ((VL_LTS_III(9, 0x13U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x12U) 
                                                                                | ((VL_LTS_III(9, 0x12U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x11U) 
                                                                                | ((VL_LTS_III(9, 0x11U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x10U) 
                                                                                | ((VL_LTS_III(9, 0x10U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xfU) 
                                                                                | ((VL_LTS_III(9, 0xfU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xeU) 
                                                                                | ((VL_LTS_III(9, 0xeU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xdU) 
                                                                                | ((VL_LTS_III(9, 0xdU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xcU) 
                                                                                | ((VL_LTS_III(9, 0xcU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xbU) 
                                                                                | ((VL_LTS_III(9, 0xbU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xaU) 
                                                                                | ((VL_LTS_III(9, 0xaU, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 9U) 
                                                                                | ((VL_LTS_III(9, 9U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 8U) 
                                                                                | ((VL_LTS_III(9, 8U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 7U) 
                                                                                | ((VL_LTS_III(9, 7U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 6U) 
                                                                                | ((VL_LTS_III(9, 6U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 5U) 
                                                                                | ((VL_LTS_III(9, 5U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 4U) 
                                                                                | ((VL_LTS_III(9, 4U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 3U) 
                                                                                | ((VL_LTS_III(9, 3U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 2U) 
                                                                                | ((VL_LTS_III(9, 2U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 1U) 
                                                                                | VL_LTS_III(9, 1U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));
    vlSelf->bf16_exp2__DOT__eo_delay[0U] = vlSelf->bf16_exp2__DOT__s2_eo_code;
    vlSelf->bf16_exp2__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_exp2__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_exp2__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
    vlSelf->bf16_exp2__DOT__u_round__DOT__final_exponent 
        = (0x1ffU & ((IData)(vlSelf->bf16_exp2__DOT__s6_poly_exp) 
                     + vlSelf->bf16_exp2__DOT__int_delay
                     [6U]));
    vlSelf->bf16_exp2__DOT__u_round__DOT__is_sub = 
        VL_GTS_III(9, 0x182U, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__final_exponent));
    vlSelf->bf16_exp2__DOT__u_round__DOT__shift_9 = 
        ((IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__is_sub)
          ? (0x1ffU & ((IData)(0x33U) + ((IData)(0x182U) 
                                         - (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__final_exponent))))
          : 0x33U);
    bf16_exp2__DOT__u_round__DOT__lsb_bit = (VL_GTS_III(32, 0x3bU, 
                                                        VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift))) 
                                             & ((0x3aU 
                                                 >= 
                                                 (0x3fU 
                                                  & (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift))) 
                                                && (1U 
                                                    & (IData)(
                                                              (vlSelf->bf16_exp2__DOT__u_round__DOT__p1_mant 
                                                               >> 
                                                               (0x3fU 
                                                                & (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))))));
    bf16_exp2__DOT__u_round__DOT__guard_bit = ((VL_LTS_III(32, 0U, 
                                                           VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift))) 
                                                & VL_GTES_III(32, 0x3bU, 
                                                              VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))) 
                                               & ((0x3aU 
                                                   >= 
                                                   (0x3fU 
                                                    & (VL_EXTENDS_II(6,9, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                       - (IData)(1U)))) 
                                                  && (1U 
                                                      & (IData)(
                                                                (vlSelf->bf16_exp2__DOT__u_round__DOT__p1_mant 
                                                                 >> 
                                                                 (0x3fU 
                                                                  & (VL_EXTENDS_II(6,9, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                     - (IData)(1U))))))));
    bf16_exp2__DOT__u_round__DOT__round_up = ((IData)(bf16_exp2__DOT__u_round__DOT__guard_bit) 
                                              & ((IData)(bf16_exp2__DOT__u_round__DOT__lsb_bit) 
                                                 | (IData)(
                                                           (0ULL 
                                                            != bf16_exp2__DOT__u_round__DOT__sticky_masked))));
    vlSelf->bf16_exp2__DOT__u_round__DOT__sum_m_ext 
        = (VL_GTS_III(32, 0x3bU, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))
            ? (0x1ffU & (IData)((0x7ffffffffffffffULL 
                                 & VL_SHIFTR_QQI(59,59,9, vlSelf->bf16_exp2__DOT__u_round__DOT__p1_mant, (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift)))))
            : 0U);
    if (bf16_exp2__DOT__u_round__DOT__round_up) {
        vlSelf->bf16_exp2__DOT__u_round__DOT__sum_m_ext 
            = (0x1ffU & ((IData)(1U) + (IData)(vlSelf->bf16_exp2__DOT__u_round__DOT__sum_m_ext)));
    }
    if ((3U == vlSelf->bf16_exp2__DOT__eo_delay[0xdU])) {
        vlSelf->bf16_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_exp2__DOT__s8_final_fp = (8U | vlSelf->bf16_exp2__DOT__s8_final_fp);
        vlSelf->bf16_exp2__DOT__s8_final_fp = (0x200000U 
                                               | vlSelf->bf16_exp2__DOT__s8_final_fp);
        vlSelf->bf16_exp2__DOT__s8_final_fp = (0x800U 
                                               | (0x3ff01fU 
                                                  & vlSelf->bf16_exp2__DOT__s8_final_fp));
    } else if ((1U == vlSelf->bf16_exp2__DOT__eo_delay
                [0xdU])) {
        vlSelf->bf16_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_exp2__DOT__s8_final_fp = (0x200fffU 
                                               & vlSelf->bf16_exp2__DOT__s8_final_fp);
        vlSelf->bf16_exp2__DOT__s8_final_fp = (0x10U 
                                               | vlSelf->bf16_exp2__DOT__s8_final_fp);
    } else if ((2U == vlSelf->bf16_exp2__DOT__eo_delay
                [0xdU])) {
        vlSelf->bf16_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_exp2__DOT__s8_final_fp = (1U | vlSelf->bf16_exp2__DOT__s8_final_fp);
    } else {
        vlSelf->bf16_exp2__DOT__s8_final_fp = vlSelf->bf16_exp2__DOT__s7_rounded_fp;
    }
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0U;
    if (VL_ONEHOT0_I(((8U & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                             << 2U)) | ((4U & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                               >> 1U)) 
                                        | ((2U & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                                  >> 1U)) 
                                           | (1U & vlSelf->bf16_exp2__DOT__s8_final_fp)))))) {
        if ((1U & vlSelf->bf16_exp2__DOT__s8_final_fp)) {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
        } else if ((4U & vlSelf->bf16_exp2__DOT__s8_final_fp)) {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
        } else if ((8U & vlSelf->bf16_exp2__DOT__s8_final_fp)) {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                = ((0U == (0x7fU & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                    >> 5U))) ? 0x40U
                    : (0x7fU & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                >> 5U)));
        } else if ((2U & vlSelf->bf16_exp2__DOT__s8_final_fp)) {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                = (0x7fU & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                            >> 5U));
        } else {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp 
                = (0x1ffU & ((IData)(0x7fU) + (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                               >> 0xcU)));
            if (VL_GTES_III(9, 0U, (IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp))) {
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
            } else if (VL_LTES_III(9, 0xffU, (IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp))) {
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
            } else {
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp 
                    = (0xffU & (IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp));
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                    = (0x7fU & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                >> 5U));
            }
        }
    }
    vlSelf->bf16_exp2__DOT__u_recompose__DOT__bf16_comb 
        = ((0x8000U & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                       >> 6U)) | (((IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp) 
                                   << 7U) | (IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out)));
}

VL_ATTR_COLD void Vbf16_exp2___024root___eval_stl(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vbf16_exp2___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vbf16_exp2___024root___eval_triggers__stl(Vbf16_exp2___024root* vlSelf);

VL_ATTR_COLD bool Vbf16_exp2___024root___eval_phase__stl(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vbf16_exp2___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vbf16_exp2___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_exp2___024root___dump_triggers__ico(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VicoTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        VL_DBG_MSGF("         'ico' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_exp2___024root___dump_triggers__act(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk or negedge rst_n)\n");
    }
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_exp2___024root___dump_triggers__nba(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk or negedge rst_n)\n");
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vbf16_exp2___024root___ctor_var_reset(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->rst_n = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tdata = VL_RAND_RESET_I(16);
    vlSelf->s_axis_tuser = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tvalid = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tready = VL_RAND_RESET_I(1);
    vlSelf->m_axis_tdata = VL_RAND_RESET_I(16);
    vlSelf->m_axis_tvalid = VL_RAND_RESET_I(1);
    vlSelf->m_axis_tready = VL_RAND_RESET_I(1);
    vlSelf->bf16_exp2__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_exp2__DOT__s2_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_exp2__DOT__s3_base2 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->bf16_exp2__DOT__exp_delay[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->bf16_exp2__DOT__s3_mant_src = VL_RAND_RESET_I(8);
    vlSelf->bf16_exp2__DOT__s3_mant_out = VL_RAND_RESET_I(31);
    vlSelf->bf16_exp2__DOT__s4_frac_part = VL_RAND_RESET_Q(38);
    vlSelf->bf16_exp2__DOT__s4_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__s5_unnorm_res = VL_RAND_RESET_Q(62);
    vlSelf->bf16_exp2__DOT__s6_norm_mant = VL_RAND_RESET_Q(59);
    vlSelf->bf16_exp2__DOT__s6_poly_exp = VL_RAND_RESET_I(9);
    for (int __Vi0 = 0; __Vi0 < 7; ++__Vi0) {
        vlSelf->bf16_exp2__DOT__int_delay[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->bf16_exp2__DOT__s7_rounded_fp = VL_RAND_RESET_I(22);
    for (int __Vi0 = 0; __Vi0 < 14; ++__Vi0) {
        vlSelf->bf16_exp2__DOT__eo_delay[__Vi0] = VL_RAND_RESET_I(2);
    }
    vlSelf->bf16_exp2__DOT__s8_final_fp = VL_RAND_RESET_I(22);
    vlSelf->bf16_exp2__DOT__bf16_out = VL_RAND_RESET_I(16);
    vlSelf->bf16_exp2__DOT__gen_axi_pipelined__DOT__vld_sr = VL_RAND_RESET_I(16);
    vlSelf->bf16_exp2__DOT__u_decompose__DOT__status_comb = VL_RAND_RESET_I(4);
    vlSelf->bf16_exp2__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    vlSelf->bf16_exp2__DOT__u_log2e_mult__DOT__mant_mult = VL_RAND_RESET_I(31);
    vlSelf->bf16_exp2__DOT__u_log2e_mult__DOT__mant_src_d = VL_RAND_RESET_I(8);
    vlSelf->bf16_exp2__DOT__u_log2e_mult__DOT__base2_d = VL_RAND_RESET_I(1);
    vlSelf->bf16_exp2__DOT__u_log2e_mult__DOT__gen_rne_reg__DOT__mant_rne_q = VL_RAND_RESET_I(31);
    vlSelf->bf16_exp2__DOT__u_unified_shift__DOT__unified_shifted = VL_RAND_RESET_Q(47);
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff = VL_RAND_RESET_Q(42);
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__frac_aligned = VL_RAND_RESET_Q(38);
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__coeff_b_stage = VL_RAND_RESET_I(21);
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__calc_res = VL_RAND_RESET_Q(62);
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[__Vi0] = VL_RAND_RESET_Q(36);
    }
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[__Vi0] = VL_RAND_RESET_Q(36);
    }
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged = VL_RAND_RESET_Q(59);
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__merged_q = VL_RAND_RESET_Q(59);
    vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__coeff_b_d = VL_RAND_RESET_I(21);
    for (int __Vi0 = 0; __Vi0 < 128; ++__Vi0) {
        vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[__Vi0] = VL_RAND_RESET_Q(42);
    }
    vlSelf->bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_normalize__DOT__msb_idx_s = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_normalize__DOT__res_s = VL_RAND_RESET_Q(62);
    vlSelf->bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb = VL_RAND_RESET_Q(59);
    vlSelf->bf16_exp2__DOT__u_round__DOT__final_exponent = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_round__DOT__is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_exp2__DOT__u_round__DOT__shift_9 = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_round__DOT__p1_final_exp = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_round__DOT__p1_is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_exp2__DOT__u_round__DOT__p1_shift = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_round__DOT__p1_mant = VL_RAND_RESET_Q(59);
    vlSelf->bf16_exp2__DOT__u_round__DOT__sum_m_ext = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_round__DOT__p2_m_ext = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_round__DOT__p2_base_exp = VL_RAND_RESET_I(9);
    vlSelf->bf16_exp2__DOT__u_round__DOT__p2_is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_exp2__DOT__u_round__DOT__rounded_comb = VL_RAND_RESET_I(22);
    vlSelf->bf16_exp2__DOT__u_recompose__DOT__bf16_comb = VL_RAND_RESET_I(16);
    vlSelf->__Vdlyvval__bf16_exp2__DOT__exp_delay__v0 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_exp2__DOT__exp_delay__v0 = 0;
    vlSelf->__Vdlyvset__bf16_exp2__DOT__exp_delay__v1 = 0;
    vlSelf->__Vdlyvval__bf16_exp2__DOT__exp_delay__v2 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_exp2__DOT__exp_delay__v2 = 0;
    vlSelf->__Vdlyvset__bf16_exp2__DOT__exp_delay__v3 = 0;
    vlSelf->__Vdlyvval__bf16_exp2__DOT__exp_delay__v4 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_exp2__DOT__exp_delay__v4 = 0;
    vlSelf->__Vdlyvset__bf16_exp2__DOT__exp_delay__v5 = 0;
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = VL_RAND_RESET_I(1);
}
