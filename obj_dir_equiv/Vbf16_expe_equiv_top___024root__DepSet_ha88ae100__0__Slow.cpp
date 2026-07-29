// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_equiv_top.h for the primary calling header

#include "Vbf16_expe_equiv_top__pch.h"
#include "Vbf16_expe_equiv_top___024root.h"

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___eval_static(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___eval_initial__TOP(Vbf16_expe_equiv_top___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___eval_initial(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_initial\n"); );
    // Body
    Vbf16_expe_equiv_top___024root___eval_initial__TOP(vlSelf);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = vlSelf->rst_n;
}

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___eval_initial__TOP(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_initial__TOP\n"); );
    // Init
    IData/*31:0*/ __Vilp;
    // Body
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0U] = 0x1b22a659157ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[1U] = 0x1b31fa59915ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[2U] = 0x1b41445a0ddULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[3U] = 0x1b50865a8b0ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[4U] = 0x1b5fba5b08dULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[5U] = 0x1b6ee65b876ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[6U] = 0x1b7e065c06aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[7U] = 0x1b8d1c5c868ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[8U] = 0x1b9c265d072ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[9U] = 0x1bab265d887ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xaU] = 0x1bba185e0a8ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xbU] = 0x1bc9005e8d3ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xcU] = 0x1bd7dc5f10aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xdU] = 0x1be6aa5f94cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xeU] = 0x1bf56c6019aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xfU] = 0x1c0422609f4ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x10U] = 0x1c12ca61258ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x11U] = 0x1c216661ac9ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x12U] = 0x1c2ff262345ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x13U] = 0x1c3e7262bceULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x14U] = 0x1c4ce463462ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x15U] = 0x1c5b4663d02ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x16U] = 0x1c699a645aeULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x17U] = 0x1c77e064e66ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x18U] = 0x1c86166572aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x19U] = 0x1c943c65ffaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1aU] = 0x1ca252668d6ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1bU] = 0x1cb058671bfULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1cU] = 0x1cbe4e67ab5ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1dU] = 0x1ccc34683b6ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1eU] = 0x1cda0a68cc4ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1fU] = 0x1ce7cc695dfULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x20U] = 0x1cf57e69f07ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x21U] = 0x1d031e6a83bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x22U] = 0x1d10ae6b17cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x23U] = 0x1d1e286bacaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x24U] = 0x1d2b926c424ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x25U] = 0x1d38e86cd8cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x26U] = 0x1d462c6d701ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x27U] = 0x1d535a6e083ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x28U] = 0x1d60766ea12ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x29U] = 0x1d6d7c6f3afULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2aU] = 0x1d7a706fd59ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2bU] = 0x1d874e70710ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2cU] = 0x1d9416710d5ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2dU] = 0x1da0c871aa7ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2eU] = 0x1dad6672487ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2fU] = 0x1db9ec72e75ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x30U] = 0x1dc65e73870ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x31U] = 0x1dd2b67427aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x32U] = 0x1ddefa74c91ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x33U] = 0x1deb24756b7ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x34U] = 0x1df736760eaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x35U] = 0x1e033276b2cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x36U] = 0x1e0f147757cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x37U] = 0x1e1adc77fdaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x38U] = 0x1e268c78a47ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x39U] = 0x1e3222794c2ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3aU] = 0x1e3da079f4cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3bU] = 0x1e49007a9e4ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3cU] = 0x1e54487b48bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3dU] = 0x1e5f747bf41ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3eU] = 0x1e6a847ca06ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3fU] = 0x1e75787d4daULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x40U] = 0x1e7ef67de60ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x41U] = 0x1e89b67e950ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x42U] = 0x1e94587f44fULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x43U] = 0x1e9edc7ff5eULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x44U] = 0x1ea94280a7cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x45U] = 0x1eb38a815a9ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x46U] = 0x1ebdb4820e6ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x47U] = 0x1ec7be82c33ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x48U] = 0x1ed1aa8378fULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x49U] = 0x1edb76842fbULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4aU] = 0x1ee52084e77ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4bU] = 0x1eeeaa85a03ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4cU] = 0x1ef8128659fULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4dU] = 0x1f015a8714bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4eU] = 0x1f0a8087d07ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4fU] = 0x1f1382888d4ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x50U] = 0x1f1c62894b1ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x51U] = 0x1f251e8a09eULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x52U] = 0x1f2db68ac9cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x53U] = 0x1f362c8b8abULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x54U] = 0x1f3e7a8c4cbULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x55U] = 0x1f45a28cf74ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x56U] = 0x1f4cac8da2bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x57U] = 0x1f54948e679ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x58U] = 0x1f5c568f2d8ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x59U] = 0x1f63f08ff48ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5aU] = 0x1f6b6490bcaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5bU] = 0x1f72b09185dULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5cU] = 0x1f79d292502ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5dU] = 0x1f80cc931b8ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5eU] = 0x1f879c93e80ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5fU] = 0x1f8e4494b59ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x60U] = 0x1f952695914ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x61U] = 0x1f9b7496612ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x62U] = 0x1fa19897323ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x63U] = 0x1fa78e98046ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x64U] = 0x1fad5a98d7bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x65U] = 0x1fb2f699ac2ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x66U] = 0x1fb8669a81cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x67U] = 0x1fbda89b588ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x68U] = 0x1fc2ba9c307ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x69U] = 0x1fc79e9d099ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6aU] = 0x1fcc089dd63ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6bU] = 0x1fd0489ea3dULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6cU] = 0x1fd4a29f806ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6dU] = 0x1fd8c8a05e1ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6eU] = 0x1fdcbea13d0ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6fU] = 0x1fe082a21d2ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x70U] = 0x1fe43aa3090ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x71U] = 0x1fe788a3e82ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x72U] = 0x1feaaea4cc0ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x73U] = 0x1feda0a5b11ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x74U] = 0x1ff05ca6976ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x75U] = 0x1ff2cea777bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x76U] = 0x1ff50ea8593ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x77U] = 0x1ff728a9433ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x78U] = 0x1ff91eaa37aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x79U] = 0x1ffabcab1ecULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7aU] = 0x1ffc28ac08eULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7bU] = 0x1ffd5cacf44ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7cU] = 0x1ffe62aded1ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7dU] = 0x1fff1eaed67ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7eU] = 0x1fffa8afce8ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7fU] = 0x1ffff2b0c81ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0U] = 0x1b22a659157ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[1U] = 0x1b31fa59915ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[2U] = 0x1b41445a0ddULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[3U] = 0x1b50865a8b0ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[4U] = 0x1b5fba5b08dULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[5U] = 0x1b6ee65b876ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[6U] = 0x1b7e065c06aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[7U] = 0x1b8d1c5c868ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[8U] = 0x1b9c265d072ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[9U] = 0x1bab265d887ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xaU] = 0x1bba185e0a8ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xbU] = 0x1bc9005e8d3ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xcU] = 0x1bd7dc5f10aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xdU] = 0x1be6aa5f94cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xeU] = 0x1bf56c6019aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0xfU] = 0x1c0422609f4ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x10U] = 0x1c12ca61258ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x11U] = 0x1c216661ac9ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x12U] = 0x1c2ff262345ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x13U] = 0x1c3e7262bceULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x14U] = 0x1c4ce463462ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x15U] = 0x1c5b4663d02ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x16U] = 0x1c699a645aeULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x17U] = 0x1c77e064e66ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x18U] = 0x1c86166572aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x19U] = 0x1c943c65ffaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1aU] = 0x1ca252668d6ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1bU] = 0x1cb058671bfULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1cU] = 0x1cbe4e67ab5ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1dU] = 0x1ccc34683b6ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1eU] = 0x1cda0a68cc4ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x1fU] = 0x1ce7cc695dfULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x20U] = 0x1cf57e69f07ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x21U] = 0x1d031e6a83bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x22U] = 0x1d10ae6b17cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x23U] = 0x1d1e286bacaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x24U] = 0x1d2b926c424ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x25U] = 0x1d38e86cd8cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x26U] = 0x1d462c6d701ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x27U] = 0x1d535a6e083ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x28U] = 0x1d60766ea12ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x29U] = 0x1d6d7c6f3afULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2aU] = 0x1d7a706fd59ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2bU] = 0x1d874e70710ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2cU] = 0x1d9416710d5ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2dU] = 0x1da0c871aa7ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2eU] = 0x1dad6672487ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x2fU] = 0x1db9ec72e75ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x30U] = 0x1dc65e73870ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x31U] = 0x1dd2b67427aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x32U] = 0x1ddefa74c91ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x33U] = 0x1deb24756b7ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x34U] = 0x1df736760eaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x35U] = 0x1e033276b2cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x36U] = 0x1e0f147757cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x37U] = 0x1e1adc77fdaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x38U] = 0x1e268c78a47ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x39U] = 0x1e3222794c2ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3aU] = 0x1e3da079f4cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3bU] = 0x1e49007a9e4ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3cU] = 0x1e54487b48bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3dU] = 0x1e5f747bf41ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3eU] = 0x1e6a847ca06ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x3fU] = 0x1e75787d4daULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x40U] = 0x1e7ef67de60ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x41U] = 0x1e89b67e950ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x42U] = 0x1e94587f44fULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x43U] = 0x1e9edc7ff5eULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x44U] = 0x1ea94280a7cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x45U] = 0x1eb38a815a9ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x46U] = 0x1ebdb4820e6ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x47U] = 0x1ec7be82c33ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x48U] = 0x1ed1aa8378fULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x49U] = 0x1edb76842fbULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4aU] = 0x1ee52084e77ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4bU] = 0x1eeeaa85a03ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4cU] = 0x1ef8128659fULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4dU] = 0x1f015a8714bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4eU] = 0x1f0a8087d07ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x4fU] = 0x1f1382888d4ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x50U] = 0x1f1c62894b1ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x51U] = 0x1f251e8a09eULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x52U] = 0x1f2db68ac9cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x53U] = 0x1f362c8b8abULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x54U] = 0x1f3e7a8c4cbULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x55U] = 0x1f45a28cf74ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x56U] = 0x1f4cac8da2bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x57U] = 0x1f54948e679ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x58U] = 0x1f5c568f2d8ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x59U] = 0x1f63f08ff48ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5aU] = 0x1f6b6490bcaULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5bU] = 0x1f72b09185dULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5cU] = 0x1f79d292502ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5dU] = 0x1f80cc931b8ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5eU] = 0x1f879c93e80ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x5fU] = 0x1f8e4494b59ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x60U] = 0x1f952695914ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x61U] = 0x1f9b7496612ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x62U] = 0x1fa19897323ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x63U] = 0x1fa78e98046ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x64U] = 0x1fad5a98d7bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x65U] = 0x1fb2f699ac2ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x66U] = 0x1fb8669a81cULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x67U] = 0x1fbda89b588ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x68U] = 0x1fc2ba9c307ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x69U] = 0x1fc79e9d099ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6aU] = 0x1fcc089dd63ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6bU] = 0x1fd0489ea3dULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6cU] = 0x1fd4a29f806ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6dU] = 0x1fd8c8a05e1ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6eU] = 0x1fdcbea13d0ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x6fU] = 0x1fe082a21d2ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x70U] = 0x1fe43aa3090ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x71U] = 0x1fe788a3e82ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x72U] = 0x1feaaea4cc0ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x73U] = 0x1feda0a5b11ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x74U] = 0x1ff05ca6976ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x75U] = 0x1ff2cea777bULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x76U] = 0x1ff50ea8593ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x77U] = 0x1ff728a9433ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x78U] = 0x1ff91eaa37aULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x79U] = 0x1ffabcab1ecULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7aU] = 0x1ffc28ac08eULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7bU] = 0x1ffd5cacf44ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7cU] = 0x1ffe62aded1ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7dU] = 0x1fff1eaed67ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7eU] = 0x1fffa8afce8ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[0x7fU] = 0x1ffff2b0c81ULL;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0U] = 0x3f80U;
    __Vilp = 1U;
    while ((__Vilp <= 0xc0U)) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[__Vilp] = 0x3f7fU;
        __Vilp = ((IData)(1U) + __Vilp);
    }
    __Vilp = 0xc1U;
    while ((__Vilp <= 0x120U)) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[__Vilp] = 0x3f7eU;
        __Vilp = ((IData)(1U) + __Vilp);
    }
    __Vilp = 0x121U;
    while ((__Vilp <= 0x161U)) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[__Vilp] = 0x3f7dU;
        __Vilp = ((IData)(1U) + __Vilp);
    }
    __Vilp = 0x162U;
    while ((__Vilp <= 0x191U)) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[__Vilp] = 0x3f7cU;
        __Vilp = ((IData)(1U) + __Vilp);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x192U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x193U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x194U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x195U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x196U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x197U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x198U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x199U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x19aU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x19bU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x19cU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x19dU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x19eU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x19fU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a0U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a1U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a2U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a3U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a4U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a5U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a6U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a7U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a8U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1a9U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1aaU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1abU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1acU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1adU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1aeU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1afU] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b0U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b1U] = 0x3f7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b2U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b3U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b4U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b5U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b6U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b7U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b8U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1b9U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1baU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1bbU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1bcU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1bdU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1beU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1bfU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c0U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c1U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c2U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c3U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c4U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c5U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c6U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c7U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c8U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1c9U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1caU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1cbU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1ccU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1cdU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1ceU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1cfU] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d0U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d1U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d2U] = 0x3f7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d3U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d4U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d5U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d6U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d7U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d8U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1d9U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1daU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1dbU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1dcU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1ddU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1deU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1dfU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e0U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e1U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e2U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e3U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e4U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e5U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e6U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e7U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e8U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1e9U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1eaU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1ebU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1ecU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1edU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1eeU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1efU] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f0U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f1U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f2U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f3U] = 0x3f79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f4U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f5U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f6U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f7U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f8U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1f9U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1faU] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1fbU] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1fcU] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1fdU] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1feU] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x1ffU] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x200U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x201U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x202U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x203U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x204U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x205U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x206U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x207U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x208U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x209U] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x20aU] = 0x3f78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x20bU] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x20cU] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x20dU] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x20eU] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x20fU] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x210U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x211U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x212U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x213U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x214U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x215U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x216U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x217U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x218U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x219U] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x21aU] = 0x3f77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x21bU] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x21cU] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x21dU] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x21eU] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x21fU] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x220U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x221U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x222U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x223U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x224U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x225U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x226U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x227U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x228U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x229U] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x22aU] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x22bU] = 0x3f76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x22cU] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x22dU] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x22eU] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x22fU] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x230U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x231U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x232U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x233U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x234U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x235U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x236U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x237U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x238U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x239U] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x23aU] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x23bU] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x23cU] = 0x3f75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x23dU] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x23eU] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x23fU] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x240U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x241U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x242U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x243U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x244U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x245U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x246U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x247U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x248U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x249U] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x24aU] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x24bU] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x24cU] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x24dU] = 0x3f74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x24eU] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x24fU] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x250U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x251U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x252U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x253U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x254U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x255U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x256U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x257U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x258U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x259U] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x25aU] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x25bU] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x25cU] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x25dU] = 0x3f73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x25eU] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x25fU] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x260U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x261U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x262U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x263U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x264U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x265U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x266U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x267U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x268U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x269U] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x26aU] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x26bU] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x26cU] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x26dU] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x26eU] = 0x3f72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x26fU] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x270U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x271U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x272U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x273U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x274U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x275U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x276U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x277U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x278U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x279U] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x27aU] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x27bU] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x27cU] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x27dU] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x27eU] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x27fU] = 0x3f71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x280U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x281U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x282U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x283U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x284U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x285U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x286U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x287U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x288U] = 0x3f70U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x289U] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x28aU] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x28bU] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x28cU] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x28dU] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x28eU] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x28fU] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x290U] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x291U] = 0x3f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x292U] = 0x3f6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x293U] = 0x3f6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x294U] = 0x3f6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x295U] = 0x3f6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x296U] = 0x3f6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x297U] = 0x3f6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x298U] = 0x3f6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x299U] = 0x3f6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x29aU] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x29bU] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x29cU] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x29dU] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x29eU] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x29fU] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a0U] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a1U] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a2U] = 0x3f6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a3U] = 0x3f6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a4U] = 0x3f6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a5U] = 0x3f6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a6U] = 0x3f6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a7U] = 0x3f6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a8U] = 0x3f6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2a9U] = 0x3f6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2aaU] = 0x3f6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2abU] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2acU] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2adU] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2aeU] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2afU] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b0U] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b1U] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b2U] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b3U] = 0x3f6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b4U] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b5U] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b6U] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b7U] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b8U] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2b9U] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2baU] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2bbU] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2bcU] = 0x3f6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2bdU] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2beU] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2bfU] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c0U] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c1U] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c2U] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c3U] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c4U] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c5U] = 0x3f69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c6U] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c7U] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c8U] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2c9U] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2caU] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2cbU] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2ccU] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2cdU] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2ceU] = 0x3f68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2cfU] = 0x3f67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d0U] = 0x3f67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d1U] = 0x3f67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d2U] = 0x3f67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d3U] = 0x3f67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d4U] = 0x3f67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d5U] = 0x3f67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d6U] = 0x3f67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d7U] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d8U] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2d9U] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2daU] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2dbU] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2dcU] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2ddU] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2deU] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2dfU] = 0x3f66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e0U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e1U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e2U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e3U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e4U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e5U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e6U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e7U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e8U] = 0x3f65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2e9U] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2eaU] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2ebU] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2ecU] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2edU] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2eeU] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2efU] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f0U] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f1U] = 0x3f64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f2U] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f3U] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f4U] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f5U] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f6U] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f7U] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f8U] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2f9U] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2faU] = 0x3f63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2fbU] = 0x3f62U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2fcU] = 0x3f62U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2fdU] = 0x3f62U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2feU] = 0x3f62U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x2ffU] = 0x3f62U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x300U] = 0x3f62U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x301U] = 0x3f62U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x302U] = 0x3f61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x303U] = 0x3f61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x304U] = 0x3f61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x305U] = 0x3f61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x306U] = 0x3f61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x307U] = 0x3f60U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x308U] = 0x3f60U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x309U] = 0x3f60U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x30aU] = 0x3f60U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x30bU] = 0x3f60U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x30cU] = 0x3f5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x30dU] = 0x3f5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x30eU] = 0x3f5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x30fU] = 0x3f5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x310U] = 0x3f5eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x311U] = 0x3f5eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x312U] = 0x3f5eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x313U] = 0x3f5eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x314U] = 0x3f5eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x315U] = 0x3f5dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x316U] = 0x3f5dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x317U] = 0x3f5dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x318U] = 0x3f5dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x319U] = 0x3f5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x31aU] = 0x3f5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x31bU] = 0x3f5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x31cU] = 0x3f5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x31dU] = 0x3f5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x31eU] = 0x3f5bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x31fU] = 0x3f5bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x320U] = 0x3f5bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x321U] = 0x3f5bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x322U] = 0x3f5bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x323U] = 0x3f5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x324U] = 0x3f5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x325U] = 0x3f5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x326U] = 0x3f5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x327U] = 0x3f59U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x328U] = 0x3f59U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x329U] = 0x3f59U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x32aU] = 0x3f59U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x32bU] = 0x3f59U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x32cU] = 0x3f58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x32dU] = 0x3f58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x32eU] = 0x3f58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x32fU] = 0x3f58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x330U] = 0x3f58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x331U] = 0x3f57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x332U] = 0x3f57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x333U] = 0x3f57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x334U] = 0x3f57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x335U] = 0x3f57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x336U] = 0x3f56U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x337U] = 0x3f56U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x338U] = 0x3f56U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x339U] = 0x3f56U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x33aU] = 0x3f55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x33bU] = 0x3f55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x33cU] = 0x3f55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x33dU] = 0x3f55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x33eU] = 0x3f55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x33fU] = 0x3f54U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x340U] = 0x3f54U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x341U] = 0x3f54U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x342U] = 0x3f54U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x343U] = 0x3f54U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x344U] = 0x3f53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x345U] = 0x3f53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x346U] = 0x3f53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x347U] = 0x3f53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x348U] = 0x3f53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x349U] = 0x3f52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x34aU] = 0x3f52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x34bU] = 0x3f52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x34cU] = 0x3f52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x34dU] = 0x3f52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x34eU] = 0x3f51U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x34fU] = 0x3f51U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x350U] = 0x3f51U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x351U] = 0x3f51U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x352U] = 0x3f51U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x353U] = 0x3f50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x354U] = 0x3f50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x355U] = 0x3f50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x356U] = 0x3f50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x357U] = 0x3f50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x358U] = 0x3f4fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x359U] = 0x3f4fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x35aU] = 0x3f4fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x35bU] = 0x3f4fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x35cU] = 0x3f4fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x35dU] = 0x3f4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x35eU] = 0x3f4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x35fU] = 0x3f4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x360U] = 0x3f4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x361U] = 0x3f4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x362U] = 0x3f4dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x363U] = 0x3f4dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x364U] = 0x3f4dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x365U] = 0x3f4dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x366U] = 0x3f4dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x367U] = 0x3f4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x368U] = 0x3f4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x369U] = 0x3f4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x36aU] = 0x3f4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x36bU] = 0x3f4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x36cU] = 0x3f4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x36dU] = 0x3f4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x36eU] = 0x3f4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x36fU] = 0x3f4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x370U] = 0x3f4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x371U] = 0x3f4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x372U] = 0x3f4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x373U] = 0x3f4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x374U] = 0x3f4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x375U] = 0x3f4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x376U] = 0x3f49U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x377U] = 0x3f49U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x378U] = 0x3f49U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x379U] = 0x3f49U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x37aU] = 0x3f49U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x37bU] = 0x3f48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x37cU] = 0x3f48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x37dU] = 0x3f48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x37eU] = 0x3f48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x37fU] = 0x3f48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x380U] = 0x3f47U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x381U] = 0x3f47U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x382U] = 0x3f47U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x383U] = 0x3f46U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x384U] = 0x3f46U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x385U] = 0x3f45U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x386U] = 0x3f45U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x387U] = 0x3f45U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x388U] = 0x3f44U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x389U] = 0x3f44U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x38aU] = 0x3f44U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x38bU] = 0x3f43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x38cU] = 0x3f43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x38dU] = 0x3f42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x38eU] = 0x3f42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x38fU] = 0x3f42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x390U] = 0x3f41U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x391U] = 0x3f41U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x392U] = 0x3f40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x393U] = 0x3f40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x394U] = 0x3f40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x395U] = 0x3f3fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x396U] = 0x3f3fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x397U] = 0x3f3fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x398U] = 0x3f3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x399U] = 0x3f3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x39aU] = 0x3f3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x39bU] = 0x3f3dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x39cU] = 0x3f3dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x39dU] = 0x3f3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x39eU] = 0x3f3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x39fU] = 0x3f3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a0U] = 0x3f3bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a1U] = 0x3f3bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a2U] = 0x3f3bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a3U] = 0x3f3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a4U] = 0x3f3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a5U] = 0x3f39U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a6U] = 0x3f39U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a7U] = 0x3f39U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a8U] = 0x3f38U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3a9U] = 0x3f38U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3aaU] = 0x3f38U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3abU] = 0x3f37U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3acU] = 0x3f37U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3adU] = 0x3f37U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3aeU] = 0x3f36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3afU] = 0x3f36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b0U] = 0x3f36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b1U] = 0x3f35U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b2U] = 0x3f35U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b3U] = 0x3f34U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b4U] = 0x3f34U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b5U] = 0x3f34U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b6U] = 0x3f33U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b7U] = 0x3f33U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b8U] = 0x3f33U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3b9U] = 0x3f32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3baU] = 0x3f32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3bbU] = 0x3f32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3bcU] = 0x3f31U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3bdU] = 0x3f31U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3beU] = 0x3f31U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3bfU] = 0x3f30U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c0U] = 0x3f30U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c1U] = 0x3f30U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c2U] = 0x3f2fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c3U] = 0x3f2fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c4U] = 0x3f2fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c5U] = 0x3f2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c6U] = 0x3f2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c7U] = 0x3f2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c8U] = 0x3f2dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3c9U] = 0x3f2dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3caU] = 0x3f2dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3cbU] = 0x3f2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3ccU] = 0x3f2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3cdU] = 0x3f2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3ceU] = 0x3f2bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3cfU] = 0x3f2bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d0U] = 0x3f2bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d1U] = 0x3f2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d2U] = 0x3f2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d3U] = 0x3f2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d4U] = 0x3f29U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d5U] = 0x3f29U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d6U] = 0x3f29U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d7U] = 0x3f28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d8U] = 0x3f28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3d9U] = 0x3f28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3daU] = 0x3f27U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3dbU] = 0x3f27U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3dcU] = 0x3f27U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3ddU] = 0x3f26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3deU] = 0x3f26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3dfU] = 0x3f26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e0U] = 0x3f25U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e1U] = 0x3f25U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e2U] = 0x3f25U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e3U] = 0x3f24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e4U] = 0x3f24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e5U] = 0x3f24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e6U] = 0x3f23U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e7U] = 0x3f23U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e8U] = 0x3f23U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3e9U] = 0x3f22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3eaU] = 0x3f22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3ebU] = 0x3f22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3ecU] = 0x3f21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3edU] = 0x3f21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3eeU] = 0x3f21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3efU] = 0x3f21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f0U] = 0x3f20U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f1U] = 0x3f20U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f2U] = 0x3f20U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f3U] = 0x3f1fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f4U] = 0x3f1fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f5U] = 0x3f1fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f6U] = 0x3f1eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f7U] = 0x3f1eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f8U] = 0x3f1eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3f9U] = 0x3f1dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3faU] = 0x3f1dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3fbU] = 0x3f1dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3fcU] = 0x3f1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3fdU] = 0x3f1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3feU] = 0x3f1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x3ffU] = 0x3f1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x400U] = 0x3f1bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x401U] = 0x3f1bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x402U] = 0x3f1aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x403U] = 0x3f19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x404U] = 0x3f19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x405U] = 0x3f18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x406U] = 0x3f18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x407U] = 0x3f17U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x408U] = 0x3f16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x409U] = 0x3f16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x40aU] = 0x3f15U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x40bU] = 0x3f15U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x40cU] = 0x3f14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x40dU] = 0x3f14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x40eU] = 0x3f13U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x40fU] = 0x3f12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x410U] = 0x3f12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x411U] = 0x3f11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x412U] = 0x3f11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x413U] = 0x3f10U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x414U] = 0x3f10U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x415U] = 0x3f0fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x416U] = 0x3f0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x417U] = 0x3f0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x418U] = 0x3f0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x419U] = 0x3f0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x41aU] = 0x3f0cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x41bU] = 0x3f0cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x41cU] = 0x3f0bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x41dU] = 0x3f0bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x41eU] = 0x3f0aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x41fU] = 0x3f0aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x420U] = 0x3f09U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x421U] = 0x3f08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x422U] = 0x3f08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x423U] = 0x3f07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x424U] = 0x3f07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x425U] = 0x3f06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x426U] = 0x3f06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x427U] = 0x3f05U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x428U] = 0x3f05U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x429U] = 0x3f04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x42aU] = 0x3f04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x42bU] = 0x3f03U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x42cU] = 0x3f03U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x42dU] = 0x3f02U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x42eU] = 0x3f02U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x42fU] = 0x3f01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x430U] = 0x3f01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x431U] = 0x3f00U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x432U] = 0x3effU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x433U] = 0x3efeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x434U] = 0x3efdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x435U] = 0x3efcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x436U] = 0x3efbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x437U] = 0x3efbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x438U] = 0x3efaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x439U] = 0x3ef9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x43aU] = 0x3ef8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x43bU] = 0x3ef7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x43cU] = 0x3ef6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x43dU] = 0x3ef5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x43eU] = 0x3ef4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x43fU] = 0x3ef3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x440U] = 0x3ef2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x441U] = 0x3ef1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x442U] = 0x3ef0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x443U] = 0x3eefU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x444U] = 0x3eeeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x445U] = 0x3eedU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x446U] = 0x3eecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x447U] = 0x3eebU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x448U] = 0x3eeaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x449U] = 0x3ee9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x44aU] = 0x3ee9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x44bU] = 0x3ee8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x44cU] = 0x3ee7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x44dU] = 0x3ee6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x44eU] = 0x3ee5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x44fU] = 0x3ee4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x450U] = 0x3ee3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x451U] = 0x3ee2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x452U] = 0x3ee1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x453U] = 0x3ee1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x454U] = 0x3ee0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x455U] = 0x3edfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x456U] = 0x3edeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x457U] = 0x3eddU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x458U] = 0x3edcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x459U] = 0x3edbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x45aU] = 0x3edaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x45bU] = 0x3edaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x45cU] = 0x3ed9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x45dU] = 0x3ed8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x45eU] = 0x3ed7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x45fU] = 0x3ed6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x460U] = 0x3ed5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x461U] = 0x3ed5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x462U] = 0x3ed4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x463U] = 0x3ed3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x464U] = 0x3ed2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x465U] = 0x3ed1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x466U] = 0x3ed0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x467U] = 0x3ed0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x468U] = 0x3ecfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x469U] = 0x3eceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x46aU] = 0x3ecdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x46bU] = 0x3eccU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x46cU] = 0x3eccU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x46dU] = 0x3ecbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x46eU] = 0x3ecaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x46fU] = 0x3ec9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x470U] = 0x3ec9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x471U] = 0x3ec8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x472U] = 0x3ec7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x473U] = 0x3ec6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x474U] = 0x3ec5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x475U] = 0x3ec5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x476U] = 0x3ec4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x477U] = 0x3ec3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x478U] = 0x3ec2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x479U] = 0x3ec2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x47aU] = 0x3ec1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x47bU] = 0x3ec0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x47cU] = 0x3ebfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x47dU] = 0x3ebfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x47eU] = 0x3ebeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x47fU] = 0x3ebdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x480U] = 0x3ebcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x481U] = 0x3ebbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x482U] = 0x3eb9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x483U] = 0x3eb8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x484U] = 0x3eb7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x485U] = 0x3eb5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x486U] = 0x3eb4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x487U] = 0x3eb2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x488U] = 0x3eb1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x489U] = 0x3eb0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x48aU] = 0x3eaeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x48bU] = 0x3eadU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x48cU] = 0x3eabU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x48dU] = 0x3eaaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x48eU] = 0x3ea9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x48fU] = 0x3ea8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x490U] = 0x3ea6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x491U] = 0x3ea5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x492U] = 0x3ea4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x493U] = 0x3ea2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x494U] = 0x3ea1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x495U] = 0x3ea0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x496U] = 0x3e9fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x497U] = 0x3e9dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x498U] = 0x3e9cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x499U] = 0x3e9bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x49aU] = 0x3e9aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x49bU] = 0x3e99U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x49cU] = 0x3e97U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x49dU] = 0x3e96U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x49eU] = 0x3e95U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x49fU] = 0x3e94U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a0U] = 0x3e93U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a1U] = 0x3e92U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a2U] = 0x3e90U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a3U] = 0x3e8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a4U] = 0x3e8eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a5U] = 0x3e8dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a6U] = 0x3e8cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a7U] = 0x3e8bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a8U] = 0x3e8aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4a9U] = 0x3e89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4aaU] = 0x3e88U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4abU] = 0x3e87U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4acU] = 0x3e86U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4adU] = 0x3e85U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4aeU] = 0x3e83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4afU] = 0x3e82U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b0U] = 0x3e81U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b1U] = 0x3e80U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b2U] = 0x3e7fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b3U] = 0x3e7dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b4U] = 0x3e7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b5U] = 0x3e79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b6U] = 0x3e77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b7U] = 0x3e75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b8U] = 0x3e73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4b9U] = 0x3e71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4baU] = 0x3e6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4bbU] = 0x3e6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4bcU] = 0x3e6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4bdU] = 0x3e6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4beU] = 0x3e68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4bfU] = 0x3e66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c0U] = 0x3e64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c1U] = 0x3e63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c2U] = 0x3e61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c3U] = 0x3e5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c4U] = 0x3e5dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c5U] = 0x3e5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c6U] = 0x3e5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c7U] = 0x3e58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c8U] = 0x3e57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4c9U] = 0x3e55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4caU] = 0x3e53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4cbU] = 0x3e52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4ccU] = 0x3e50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4cdU] = 0x3e4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4ceU] = 0x3e4dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4cfU] = 0x3e4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d0U] = 0x3e4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d1U] = 0x3e48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d2U] = 0x3e47U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d3U] = 0x3e45U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d4U] = 0x3e43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d5U] = 0x3e42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d6U] = 0x3e40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d7U] = 0x3e3fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d8U] = 0x3e3dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4d9U] = 0x3e3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4daU] = 0x3e3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4dbU] = 0x3e39U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4dcU] = 0x3e38U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4ddU] = 0x3e36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4deU] = 0x3e35U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4dfU] = 0x3e33U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e0U] = 0x3e32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e1U] = 0x3e31U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e2U] = 0x3e2fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e3U] = 0x3e2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e4U] = 0x3e2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e5U] = 0x3e2bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e6U] = 0x3e2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e7U] = 0x3e28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e8U] = 0x3e27U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4e9U] = 0x3e26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4eaU] = 0x3e25U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4ebU] = 0x3e23U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4ecU] = 0x3e22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4edU] = 0x3e21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4eeU] = 0x3e20U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4efU] = 0x3e1eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f0U] = 0x3e1dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f1U] = 0x3e1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f2U] = 0x3e1bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f3U] = 0x3e19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f4U] = 0x3e18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f5U] = 0x3e17U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f6U] = 0x3e16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f7U] = 0x3e15U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f8U] = 0x3e14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4f9U] = 0x3e12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4faU] = 0x3e11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4fbU] = 0x3e10U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4fcU] = 0x3e0fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4fdU] = 0x3e0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4feU] = 0x3e0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x4ffU] = 0x3e0cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x500U] = 0x3e0bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x501U] = 0x3e08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x502U] = 0x3e06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x503U] = 0x3e04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x504U] = 0x3e02U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x505U] = 0x3e00U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x506U] = 0x3dfcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x507U] = 0x3df8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x508U] = 0x3df5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x509U] = 0x3df1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x50aU] = 0x3dedU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x50bU] = 0x3de9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x50cU] = 0x3de6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x50dU] = 0x3de2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x50eU] = 0x3ddfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x50fU] = 0x3ddbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x510U] = 0x3dd8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x511U] = 0x3dd5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x512U] = 0x3dd1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x513U] = 0x3dceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x514U] = 0x3dcbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x515U] = 0x3dc8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x516U] = 0x3dc5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x517U] = 0x3dc1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x518U] = 0x3dbeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x519U] = 0x3dbcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x51aU] = 0x3db9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x51bU] = 0x3db6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x51cU] = 0x3db3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x51dU] = 0x3db0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x51eU] = 0x3dadU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x51fU] = 0x3dabU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x520U] = 0x3da8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x521U] = 0x3da6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x522U] = 0x3da3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x523U] = 0x3da0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x524U] = 0x3d9eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x525U] = 0x3d9bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x526U] = 0x3d99U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x527U] = 0x3d97U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x528U] = 0x3d94U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x529U] = 0x3d92U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x52aU] = 0x3d90U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x52bU] = 0x3d8eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x52cU] = 0x3d8bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x52dU] = 0x3d89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x52eU] = 0x3d87U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x52fU] = 0x3d85U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x530U] = 0x3d83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x531U] = 0x3d81U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x532U] = 0x3d7eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x533U] = 0x3d7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x534U] = 0x3d76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x535U] = 0x3d72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x536U] = 0x3d6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x537U] = 0x3d6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x538U] = 0x3d67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x539U] = 0x3d63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x53aU] = 0x3d60U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x53bU] = 0x3d5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x53cU] = 0x3d59U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x53dU] = 0x3d56U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x53eU] = 0x3d52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x53fU] = 0x3d4fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x540U] = 0x3d4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x541U] = 0x3d49U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x542U] = 0x3d46U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x543U] = 0x3d43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x544U] = 0x3d40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x545U] = 0x3d3dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x546U] = 0x3d3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x547U] = 0x3d37U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x548U] = 0x3d34U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x549U] = 0x3d31U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x54aU] = 0x3d2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x54bU] = 0x3d2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x54cU] = 0x3d29U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x54dU] = 0x3d26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x54eU] = 0x3d24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x54fU] = 0x3d21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x550U] = 0x3d1fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x551U] = 0x3d1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x552U] = 0x3d1aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x553U] = 0x3d18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x554U] = 0x3d15U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x555U] = 0x3d13U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x556U] = 0x3d11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x557U] = 0x3d0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x558U] = 0x3d0cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x559U] = 0x3d0aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x55aU] = 0x3d08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x55bU] = 0x3d06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x55cU] = 0x3d04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x55dU] = 0x3d02U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x55eU] = 0x3cffU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x55fU] = 0x3cfbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x560U] = 0x3cf7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x561U] = 0x3cf4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x562U] = 0x3cf0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x563U] = 0x3cecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x564U] = 0x3ce8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x565U] = 0x3ce5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x566U] = 0x3ce1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x567U] = 0x3cdeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x568U] = 0x3cdaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x569U] = 0x3cd7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x56aU] = 0x3cd4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x56bU] = 0x3cd0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x56cU] = 0x3ccdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x56dU] = 0x3ccaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x56eU] = 0x3cc7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x56fU] = 0x3cc4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x570U] = 0x3cc1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x571U] = 0x3cbeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x572U] = 0x3cbbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x573U] = 0x3cb8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x574U] = 0x3cb5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x575U] = 0x3cb2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x576U] = 0x3cafU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x577U] = 0x3cadU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x578U] = 0x3caaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x579U] = 0x3ca7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x57aU] = 0x3ca5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x57bU] = 0x3ca2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x57cU] = 0x3ca0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x57dU] = 0x3c9dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x57eU] = 0x3c9bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x57fU] = 0x3c98U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x580U] = 0x3c96U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x581U] = 0x3c91U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x582U] = 0x3c8dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x583U] = 0x3c89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x584U] = 0x3c84U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x585U] = 0x3c80U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x586U] = 0x3c79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x587U] = 0x3c71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x588U] = 0x3c6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x589U] = 0x3c63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x58aU] = 0x3c5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x58bU] = 0x3c55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x58cU] = 0x3c4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x58dU] = 0x3c48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x58eU] = 0x3c42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x58fU] = 0x3c3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x590U] = 0x3c36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x591U] = 0x3c30U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x592U] = 0x3c2bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x593U] = 0x3c26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x594U] = 0x3c21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x595U] = 0x3c1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x596U] = 0x3c17U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x597U] = 0x3c12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x598U] = 0x3c0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x599U] = 0x3c09U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x59aU] = 0x3c05U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x59bU] = 0x3c01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x59cU] = 0x3bfaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x59dU] = 0x3bf2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x59eU] = 0x3bebU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x59fU] = 0x3be4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a0U] = 0x3bddU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a1U] = 0x3bd6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a2U] = 0x3bcfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a3U] = 0x3bc9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a4U] = 0x3bc3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a5U] = 0x3bbdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a6U] = 0x3bb7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a7U] = 0x3bb1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a8U] = 0x3bacU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5a9U] = 0x3ba7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5aaU] = 0x3ba2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5abU] = 0x3b9dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5acU] = 0x3b98U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5adU] = 0x3b93U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5aeU] = 0x3b8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5afU] = 0x3b8aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b0U] = 0x3b86U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b1U] = 0x3b82U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b2U] = 0x3b7cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b3U] = 0x3b74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b4U] = 0x3b6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b5U] = 0x3b65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b6U] = 0x3b5eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b7U] = 0x3b57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b8U] = 0x3b51U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5b9U] = 0x3b4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5baU] = 0x3b44U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5bbU] = 0x3b3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5bcU] = 0x3b38U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5bdU] = 0x3b32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5beU] = 0x3b2dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5bfU] = 0x3b28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c0U] = 0x3b22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c1U] = 0x3b1dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c2U] = 0x3b19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c3U] = 0x3b14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c4U] = 0x3b0fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c5U] = 0x3b0bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c6U] = 0x3b07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c7U] = 0x3b03U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c8U] = 0x3afdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5c9U] = 0x3af5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5caU] = 0x3aeeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5cbU] = 0x3ae6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5ccU] = 0x3adfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5cdU] = 0x3ad8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5ceU] = 0x3ad2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5cfU] = 0x3acbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d0U] = 0x3ac5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d1U] = 0x3abfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d2U] = 0x3ab9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d3U] = 0x3ab3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d4U] = 0x3aaeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d5U] = 0x3aa9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d6U] = 0x3aa3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d7U] = 0x3a9eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d8U] = 0x3a99U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5d9U] = 0x3a95U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5daU] = 0x3a90U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5dbU] = 0x3a8cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5dcU] = 0x3a87U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5ddU] = 0x3a83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5deU] = 0x3a7eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5dfU] = 0x3a77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e0U] = 0x3a6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e1U] = 0x3a68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e2U] = 0x3a61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e3U] = 0x3a5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e4U] = 0x3a53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e5U] = 0x3a4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e6U] = 0x3a46U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e7U] = 0x3a40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e8U] = 0x3a3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5e9U] = 0x3a34U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5eaU] = 0x3a2fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5ebU] = 0x3a2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5ecU] = 0x3a24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5edU] = 0x3a1fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5eeU] = 0x3a1aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5efU] = 0x3a16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f0U] = 0x3a11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f1U] = 0x3a0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f2U] = 0x3a08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f3U] = 0x3a04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f4U] = 0x3a00U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f5U] = 0x39f8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f6U] = 0x39f0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f7U] = 0x39e9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f8U] = 0x39e2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5f9U] = 0x39dbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5faU] = 0x39d4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5fbU] = 0x39ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5fcU] = 0x39c7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5fdU] = 0x39c1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5feU] = 0x39bbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x5ffU] = 0x39b5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x600U] = 0x39b0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x601U] = 0x39a5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x602U] = 0x399bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x603U] = 0x3992U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x604U] = 0x3989U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x605U] = 0x3981U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x606U] = 0x3972U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x607U] = 0x3963U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x608U] = 0x3955U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x609U] = 0x3948U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x60aU] = 0x393cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x60bU] = 0x3931U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x60cU] = 0x3926U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x60dU] = 0x391cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x60eU] = 0x3913U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x60fU] = 0x390aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x610U] = 0x3901U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x611U] = 0x38f3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x612U] = 0x38e4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x613U] = 0x38d7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x614U] = 0x38caU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x615U] = 0x38bdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x616U] = 0x38b2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x617U] = 0x38a7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x618U] = 0x389dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x619U] = 0x3893U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x61aU] = 0x388bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x61bU] = 0x3882U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x61cU] = 0x3875U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x61dU] = 0x3866U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x61eU] = 0x3858U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x61fU] = 0x384bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x620U] = 0x383eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x621U] = 0x3833U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x622U] = 0x3828U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x623U] = 0x381eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x624U] = 0x3814U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x625U] = 0x380bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x626U] = 0x3803U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x627U] = 0x37f6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x628U] = 0x37e7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x629U] = 0x37d9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x62aU] = 0x37ccU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x62bU] = 0x37bfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x62cU] = 0x37b4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x62dU] = 0x37a9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x62eU] = 0x379fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x62fU] = 0x3795U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x630U] = 0x378cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x631U] = 0x3784U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x632U] = 0x3777U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x633U] = 0x3768U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x634U] = 0x375aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x635U] = 0x374dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x636U] = 0x3741U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x637U] = 0x3735U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x638U] = 0x372aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x639U] = 0x3720U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x63aU] = 0x3716U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x63bU] = 0x370dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x63cU] = 0x3704U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x63dU] = 0x36f9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x63eU] = 0x36eaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x63fU] = 0x36dbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x640U] = 0x36ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x641U] = 0x36c2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x642U] = 0x36b6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x643U] = 0x36abU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x644U] = 0x36a1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x645U] = 0x3697U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x646U] = 0x368eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x647U] = 0x3685U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x648U] = 0x367aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x649U] = 0x366bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x64aU] = 0x365dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x64bU] = 0x364fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x64cU] = 0x3643U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x64dU] = 0x3637U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x64eU] = 0x362cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x64fU] = 0x3621U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x650U] = 0x3618U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x651U] = 0x360eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x652U] = 0x3606U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x653U] = 0x35fcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x654U] = 0x35ecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x655U] = 0x35deU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x656U] = 0x35d1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x657U] = 0x35c4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x658U] = 0x35b8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x659U] = 0x35adU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x65aU] = 0x35a2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x65bU] = 0x3599U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x65cU] = 0x358fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x65dU] = 0x3587U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x65eU] = 0x357dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x65fU] = 0x356eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x660U] = 0x355fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x661U] = 0x3552U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x662U] = 0x3545U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x663U] = 0x3539U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x664U] = 0x352eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x665U] = 0x3523U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x666U] = 0x3519U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x667U] = 0x3510U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x668U] = 0x3507U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x669U] = 0x34feU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x66aU] = 0x34efU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x66bU] = 0x34e0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x66cU] = 0x34d3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x66dU] = 0x34c6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x66eU] = 0x34baU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x66fU] = 0x34afU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x670U] = 0x34a4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x671U] = 0x349aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x672U] = 0x3491U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x673U] = 0x3488U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x674U] = 0x3480U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x675U] = 0x3470U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x676U] = 0x3462U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x677U] = 0x3454U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x678U] = 0x3447U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x679U] = 0x343bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x67aU] = 0x3430U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x67bU] = 0x3425U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x67cU] = 0x341bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x67dU] = 0x3412U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x67eU] = 0x3409U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x67fU] = 0x3401U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x680U] = 0x33f2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x681U] = 0x33d5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x682U] = 0x33bcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x683U] = 0x33a6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x684U] = 0x3393U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x685U] = 0x3381U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x686U] = 0x3364U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x687U] = 0x3349U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x688U] = 0x3332U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x689U] = 0x331dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x68aU] = 0x330aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x68bU] = 0x32f4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x68cU] = 0x32d8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x68dU] = 0x32beU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x68eU] = 0x32a8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x68fU] = 0x3294U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x690U] = 0x3283U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x691U] = 0x3267U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x692U] = 0x324cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x693U] = 0x3234U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x694U] = 0x321fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x695U] = 0x320cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x696U] = 0x31f7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x697U] = 0x31daU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x698U] = 0x31c1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x699U] = 0x31aaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x69aU] = 0x3196U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x69bU] = 0x3184U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x69cU] = 0x316aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x69dU] = 0x314eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x69eU] = 0x3136U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x69fU] = 0x3121U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a0U] = 0x310eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a1U] = 0x30faU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a2U] = 0x30ddU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a3U] = 0x30c3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a4U] = 0x30acU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a5U] = 0x3098U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a6U] = 0x3086U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a7U] = 0x306cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a8U] = 0x3050U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6a9U] = 0x3038U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6aaU] = 0x3022U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6abU] = 0x300fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6acU] = 0x2ffdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6adU] = 0x2fdfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6aeU] = 0x2fc5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6afU] = 0x2faeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b0U] = 0x2f99U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b1U] = 0x2f87U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b2U] = 0x2f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b3U] = 0x2f53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b4U] = 0x2f3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b5U] = 0x2f24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b6U] = 0x2f11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b7U] = 0x2f00U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b8U] = 0x2ee2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6b9U] = 0x2ec7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6baU] = 0x2eb0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6bbU] = 0x2e9bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6bcU] = 0x2e89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6bdU] = 0x2e72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6beU] = 0x2e55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6bfU] = 0x2e3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c0U] = 0x2e26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c1U] = 0x2e13U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c2U] = 0x2e01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c3U] = 0x2de4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c4U] = 0x2dc9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c5U] = 0x2db2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c6U] = 0x2d9dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c7U] = 0x2d8aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c8U] = 0x2d74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6c9U] = 0x2d58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6caU] = 0x2d3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6cbU] = 0x2d28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6ccU] = 0x2d14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6cdU] = 0x2d03U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6ceU] = 0x2ce7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6cfU] = 0x2cccU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d0U] = 0x2cb4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d1U] = 0x2c9fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d2U] = 0x2c8cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d3U] = 0x2c77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d4U] = 0x2c5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d5U] = 0x2c40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d6U] = 0x2c2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d7U] = 0x2c16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d8U] = 0x2c04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6d9U] = 0x2be9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6daU] = 0x2bceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6dbU] = 0x2bb6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6dcU] = 0x2ba0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6ddU] = 0x2b8eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6deU] = 0x2b7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6dfU] = 0x2b5dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e0U] = 0x2b43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e1U] = 0x2b2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e2U] = 0x2b18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e3U] = 0x2b06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e4U] = 0x2aecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e5U] = 0x2ad0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e6U] = 0x2ab8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e7U] = 0x2aa2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e8U] = 0x2a8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6e9U] = 0x2a7dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6eaU] = 0x2a5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6ebU] = 0x2a45U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6ecU] = 0x2a2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6edU] = 0x2a19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6eeU] = 0x2a07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6efU] = 0x29efU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f0U] = 0x29d3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f1U] = 0x29baU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f2U] = 0x29a4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f3U] = 0x2991U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f4U] = 0x2980U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f5U] = 0x2962U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f6U] = 0x2947U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f7U] = 0x2930U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f8U] = 0x291bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6f9U] = 0x2909U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6faU] = 0x28f1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6fbU] = 0x28d5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6fcU] = 0x28bcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6fdU] = 0x28a6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6feU] = 0x2892U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x6ffU] = 0x2881U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x700U] = 0x2864U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x701U] = 0x2832U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x702U] = 0x280aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x703U] = 0x27d8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x704U] = 0x27a8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x705U] = 0x2783U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x706U] = 0x274cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x707U] = 0x271fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x708U] = 0x26f7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x709U] = 0x26c0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x70aU] = 0x2696U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x70bU] = 0x2669U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x70cU] = 0x2636U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x70dU] = 0x260eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x70eU] = 0x25dcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x70fU] = 0x25acU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x710U] = 0x2586U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x711U] = 0x2550U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x712U] = 0x2522U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x713U] = 0x24fdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x714U] = 0x24c5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x715U] = 0x2499U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x716U] = 0x246fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x717U] = 0x243aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x718U] = 0x2411U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x719U] = 0x23e1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x71aU] = 0x23b0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x71bU] = 0x2389U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x71cU] = 0x2355U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x71dU] = 0x2326U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x71eU] = 0x2301U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x71fU] = 0x22c9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x720U] = 0x229dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x721U] = 0x2274U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x722U] = 0x223eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x723U] = 0x2214U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x724U] = 0x21e7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x725U] = 0x21b4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x726U] = 0x218cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x727U] = 0x215aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x728U] = 0x212aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x729U] = 0x2104U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x72aU] = 0x20ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x72bU] = 0x20a0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x72cU] = 0x207aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x72dU] = 0x2042U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x72eU] = 0x2017U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x72fU] = 0x1fecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x730U] = 0x1fb8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x731U] = 0x1f8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x732U] = 0x1f5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x733U] = 0x1f2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x734U] = 0x1f07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x735U] = 0x1ed3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x736U] = 0x1ea4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x737U] = 0x1e7fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x738U] = 0x1e47U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x739U] = 0x1e1bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x73aU] = 0x1df1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x73bU] = 0x1dbcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x73cU] = 0x1d92U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x73dU] = 0x1d64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x73eU] = 0x1d32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x73fU] = 0x1d0aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x740U] = 0x1cd7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x741U] = 0x1ca8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x742U] = 0x1c83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x743U] = 0x1c4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x744U] = 0x1c1eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x745U] = 0x1bf7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x746U] = 0x1bc0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x747U] = 0x1b96U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x748U] = 0x1b69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x749U] = 0x1b36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x74aU] = 0x1b0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x74bU] = 0x1adcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x74cU] = 0x1aacU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x74dU] = 0x1a86U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x74eU] = 0x1a50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x74fU] = 0x1a22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x750U] = 0x19fcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x751U] = 0x19c5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x752U] = 0x1999U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x753U] = 0x196eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x754U] = 0x193aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x755U] = 0x1911U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x756U] = 0x18e1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x757U] = 0x18afU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x758U] = 0x1889U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x759U] = 0x1855U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x75aU] = 0x1826U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x75bU] = 0x1801U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x75cU] = 0x17c9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x75dU] = 0x179dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x75eU] = 0x1774U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x75fU] = 0x173eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x760U] = 0x1714U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x761U] = 0x16e6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x762U] = 0x16b3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x763U] = 0x168cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x764U] = 0x165aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x765U] = 0x162aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x766U] = 0x1604U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x767U] = 0x15ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x768U] = 0x15a0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x769U] = 0x157aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x76aU] = 0x1542U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x76bU] = 0x1517U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x76cU] = 0x14ecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x76dU] = 0x14b8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x76eU] = 0x148fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x76fU] = 0x145fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x770U] = 0x142dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x771U] = 0x1407U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x772U] = 0x13d2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x773U] = 0x13a4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x774U] = 0x137fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x775U] = 0x1347U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x776U] = 0x131bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x777U] = 0x12f1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x778U] = 0x12bcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x779U] = 0x1292U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x77aU] = 0x1264U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x77bU] = 0x1231U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x77cU] = 0x120aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x77dU] = 0x11d7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x77eU] = 0x11a8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x77fU] = 0x1183U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x780U] = 0x114bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x781U] = 0x10f7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x782U] = 0x1096U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x783U] = 0x1035U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x784U] = 0xfdcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x785U] = 0xf86U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x786U] = 0xf22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x787U] = 0xec4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x788U] = 0xe6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x789U] = 0xe11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x78aU] = 0xdafU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x78bU] = 0xd55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x78cU] = 0xd01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x78dU] = 0xc9cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x78eU] = 0xc3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x78fU] = 0xbe6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x790U] = 0xb8cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x791U] = 0xb29U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x792U] = 0xaceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x793U] = 0xa79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x794U] = 0xa17U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x795U] = 0x9b7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x796U] = 0x95fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x797U] = 0x907U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x798U] = 0x8a4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x799U] = 0x847U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x79aU] = 0x7f1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x79bU] = 0x792U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x79cU] = 0x731U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x79dU] = 0x6d7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x79eU] = 0x682U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x79fU] = 0x61eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a0U] = 0x5c0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a1U] = 0x569U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a2U] = 0x50dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a3U] = 0x4abU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a4U] = 0x450U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a5U] = 0x3fcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a6U] = 0x399U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a7U] = 0x339U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a8U] = 0x2e1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7a9U] = 0x288U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7aaU] = 0x226U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7abU] = 0x1c9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7acU] = 0x174U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7adU] = 0x114U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7aeU] = 0xb3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7afU] = 0x6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b0U] = 0x42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b1U] = 0x28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b2U] = 0x18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b3U] = 0xfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b4U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b5U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b6U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b7U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b8U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[0x7b9U] = 1U;
    __Vilp = 0x7baU;
    while ((__Vilp <= 0x87fU)) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[__Vilp] = 0U;
        __Vilp = ((IData)(1U) + __Vilp);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0U] = 0x3f1bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[1U] = 0x3f1bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[2U] = 0x3f1aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[3U] = 0x3f19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[4U] = 0x3f19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[5U] = 0x3f18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[6U] = 0x3f18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[7U] = 0x3f17U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[8U] = 0x3f16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[9U] = 0x3f16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xaU] = 0x3f15U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xbU] = 0x3f15U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xcU] = 0x3f14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xdU] = 0x3f14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xeU] = 0x3f13U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xfU] = 0x3f12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x10U] = 0x3f12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x11U] = 0x3f11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x12U] = 0x3f11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x13U] = 0x3f10U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x14U] = 0x3f10U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x15U] = 0x3f0fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x16U] = 0x3f0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x17U] = 0x3f0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x18U] = 0x3f0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x19U] = 0x3f0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1aU] = 0x3f0cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1bU] = 0x3f0cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1cU] = 0x3f0bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1dU] = 0x3f0bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1eU] = 0x3f0aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1fU] = 0x3f0aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x20U] = 0x3f09U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x21U] = 0x3f08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x22U] = 0x3f08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x23U] = 0x3f07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x24U] = 0x3f07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x25U] = 0x3f06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x26U] = 0x3f06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x27U] = 0x3f05U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x28U] = 0x3f05U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x29U] = 0x3f04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2aU] = 0x3f04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2bU] = 0x3f03U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2cU] = 0x3f03U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2dU] = 0x3f02U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2eU] = 0x3f02U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2fU] = 0x3f01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x30U] = 0x3f01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x31U] = 0x3f00U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x32U] = 0x3effU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x33U] = 0x3efeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x34U] = 0x3efdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x35U] = 0x3efcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x36U] = 0x3efbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x37U] = 0x3efbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x38U] = 0x3efaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x39U] = 0x3ef9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3aU] = 0x3ef8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3bU] = 0x3ef7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3cU] = 0x3ef6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3dU] = 0x3ef5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3eU] = 0x3ef4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3fU] = 0x3ef3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x40U] = 0x3ef2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x41U] = 0x3ef1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x42U] = 0x3ef0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x43U] = 0x3eefU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x44U] = 0x3eeeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x45U] = 0x3eedU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x46U] = 0x3eecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x47U] = 0x3eebU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x48U] = 0x3eeaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x49U] = 0x3ee9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x4aU] = 0x3ee9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x4bU] = 0x3ee8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x4cU] = 0x3ee7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x4dU] = 0x3ee6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x4eU] = 0x3ee5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x4fU] = 0x3ee4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x50U] = 0x3ee3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x51U] = 0x3ee2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x52U] = 0x3ee1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x53U] = 0x3ee1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x54U] = 0x3ee0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x55U] = 0x3edfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x56U] = 0x3edeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x57U] = 0x3eddU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x58U] = 0x3edcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x59U] = 0x3edbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x5aU] = 0x3edaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x5bU] = 0x3edaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x5cU] = 0x3ed9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x5dU] = 0x3ed8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x5eU] = 0x3ed7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x5fU] = 0x3ed6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x60U] = 0x3ed5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x61U] = 0x3ed5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x62U] = 0x3ed4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x63U] = 0x3ed3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x64U] = 0x3ed2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x65U] = 0x3ed1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x66U] = 0x3ed0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x67U] = 0x3ed0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x68U] = 0x3ecfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x69U] = 0x3eceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x6aU] = 0x3ecdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x6bU] = 0x3eccU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x6cU] = 0x3eccU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x6dU] = 0x3ecbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x6eU] = 0x3ecaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x6fU] = 0x3ec9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x70U] = 0x3ec9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x71U] = 0x3ec8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x72U] = 0x3ec7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x73U] = 0x3ec6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x74U] = 0x3ec5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x75U] = 0x3ec5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x76U] = 0x3ec4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x77U] = 0x3ec3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x78U] = 0x3ec2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x79U] = 0x3ec2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x7aU] = 0x3ec1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x7bU] = 0x3ec0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x7cU] = 0x3ebfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x7dU] = 0x3ebfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x7eU] = 0x3ebeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x7fU] = 0x3ebdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x80U] = 0x3ebcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x81U] = 0x3ebbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x82U] = 0x3eb9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x83U] = 0x3eb8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x84U] = 0x3eb7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x85U] = 0x3eb5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x86U] = 0x3eb4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x87U] = 0x3eb2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x88U] = 0x3eb1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x89U] = 0x3eb0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x8aU] = 0x3eaeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x8bU] = 0x3eadU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x8cU] = 0x3eabU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x8dU] = 0x3eaaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x8eU] = 0x3ea9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x8fU] = 0x3ea8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x90U] = 0x3ea6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x91U] = 0x3ea5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x92U] = 0x3ea4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x93U] = 0x3ea2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x94U] = 0x3ea1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x95U] = 0x3ea0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x96U] = 0x3e9fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x97U] = 0x3e9dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x98U] = 0x3e9cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x99U] = 0x3e9bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x9aU] = 0x3e9aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x9bU] = 0x3e99U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x9cU] = 0x3e97U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x9dU] = 0x3e96U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x9eU] = 0x3e95U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x9fU] = 0x3e94U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa0U] = 0x3e93U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa1U] = 0x3e92U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa2U] = 0x3e90U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa3U] = 0x3e8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa4U] = 0x3e8eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa5U] = 0x3e8dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa6U] = 0x3e8cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa7U] = 0x3e8bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa8U] = 0x3e8aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xa9U] = 0x3e89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xaaU] = 0x3e88U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xabU] = 0x3e87U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xacU] = 0x3e86U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xadU] = 0x3e85U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xaeU] = 0x3e83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xafU] = 0x3e82U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb0U] = 0x3e81U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb1U] = 0x3e80U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb2U] = 0x3e7fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb3U] = 0x3e7dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb4U] = 0x3e7bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb5U] = 0x3e79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb6U] = 0x3e77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb7U] = 0x3e75U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb8U] = 0x3e73U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xb9U] = 0x3e71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xbaU] = 0x3e6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xbbU] = 0x3e6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xbcU] = 0x3e6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xbdU] = 0x3e6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xbeU] = 0x3e68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xbfU] = 0x3e66U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc0U] = 0x3e64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc1U] = 0x3e63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc2U] = 0x3e61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc3U] = 0x3e5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc4U] = 0x3e5dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc5U] = 0x3e5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc6U] = 0x3e5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc7U] = 0x3e58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc8U] = 0x3e57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xc9U] = 0x3e55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xcaU] = 0x3e53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xcbU] = 0x3e52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xccU] = 0x3e50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xcdU] = 0x3e4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xceU] = 0x3e4dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xcfU] = 0x3e4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd0U] = 0x3e4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd1U] = 0x3e48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd2U] = 0x3e47U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd3U] = 0x3e45U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd4U] = 0x3e43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd5U] = 0x3e42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd6U] = 0x3e40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd7U] = 0x3e3fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd8U] = 0x3e3dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xd9U] = 0x3e3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xdaU] = 0x3e3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xdbU] = 0x3e39U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xdcU] = 0x3e38U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xddU] = 0x3e36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xdeU] = 0x3e35U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xdfU] = 0x3e33U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe0U] = 0x3e32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe1U] = 0x3e31U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe2U] = 0x3e2fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe3U] = 0x3e2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe4U] = 0x3e2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe5U] = 0x3e2bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe6U] = 0x3e2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe7U] = 0x3e28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe8U] = 0x3e27U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xe9U] = 0x3e26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xeaU] = 0x3e25U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xebU] = 0x3e23U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xecU] = 0x3e22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xedU] = 0x3e21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xeeU] = 0x3e20U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xefU] = 0x3e1eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf0U] = 0x3e1dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf1U] = 0x3e1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf2U] = 0x3e1bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf3U] = 0x3e19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf4U] = 0x3e18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf5U] = 0x3e17U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf6U] = 0x3e16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf7U] = 0x3e15U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf8U] = 0x3e14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xf9U] = 0x3e12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xfaU] = 0x3e11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xfbU] = 0x3e10U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xfcU] = 0x3e0fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xfdU] = 0x3e0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xfeU] = 0x3e0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0xffU] = 0x3e0cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x100U] = 0x3e0bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x101U] = 0x3e08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x102U] = 0x3e06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x103U] = 0x3e04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x104U] = 0x3e02U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x105U] = 0x3e00U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x106U] = 0x3dfcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x107U] = 0x3df8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x108U] = 0x3df5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x109U] = 0x3df1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x10aU] = 0x3dedU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x10bU] = 0x3de9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x10cU] = 0x3de6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x10dU] = 0x3de2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x10eU] = 0x3ddfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x10fU] = 0x3ddbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x110U] = 0x3dd8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x111U] = 0x3dd5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x112U] = 0x3dd1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x113U] = 0x3dceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x114U] = 0x3dcbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x115U] = 0x3dc8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x116U] = 0x3dc5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x117U] = 0x3dc1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x118U] = 0x3dbeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x119U] = 0x3dbcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x11aU] = 0x3db9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x11bU] = 0x3db6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x11cU] = 0x3db3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x11dU] = 0x3db0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x11eU] = 0x3dadU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x11fU] = 0x3dabU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x120U] = 0x3da8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x121U] = 0x3da6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x122U] = 0x3da3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x123U] = 0x3da0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x124U] = 0x3d9eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x125U] = 0x3d9bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x126U] = 0x3d99U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x127U] = 0x3d97U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x128U] = 0x3d94U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x129U] = 0x3d92U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x12aU] = 0x3d90U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x12bU] = 0x3d8eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x12cU] = 0x3d8bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x12dU] = 0x3d89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x12eU] = 0x3d87U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x12fU] = 0x3d85U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x130U] = 0x3d83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x131U] = 0x3d81U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x132U] = 0x3d7eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x133U] = 0x3d7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x134U] = 0x3d76U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x135U] = 0x3d72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x136U] = 0x3d6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x137U] = 0x3d6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x138U] = 0x3d67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x139U] = 0x3d63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x13aU] = 0x3d60U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x13bU] = 0x3d5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x13cU] = 0x3d59U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x13dU] = 0x3d56U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x13eU] = 0x3d52U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x13fU] = 0x3d4fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x140U] = 0x3d4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x141U] = 0x3d49U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x142U] = 0x3d46U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x143U] = 0x3d43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x144U] = 0x3d40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x145U] = 0x3d3dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x146U] = 0x3d3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x147U] = 0x3d37U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x148U] = 0x3d34U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x149U] = 0x3d31U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x14aU] = 0x3d2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x14bU] = 0x3d2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x14cU] = 0x3d29U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x14dU] = 0x3d26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x14eU] = 0x3d24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x14fU] = 0x3d21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x150U] = 0x3d1fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x151U] = 0x3d1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x152U] = 0x3d1aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x153U] = 0x3d18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x154U] = 0x3d15U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x155U] = 0x3d13U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x156U] = 0x3d11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x157U] = 0x3d0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x158U] = 0x3d0cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x159U] = 0x3d0aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x15aU] = 0x3d08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x15bU] = 0x3d06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x15cU] = 0x3d04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x15dU] = 0x3d02U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x15eU] = 0x3cffU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x15fU] = 0x3cfbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x160U] = 0x3cf7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x161U] = 0x3cf4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x162U] = 0x3cf0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x163U] = 0x3cecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x164U] = 0x3ce8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x165U] = 0x3ce5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x166U] = 0x3ce1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x167U] = 0x3cdeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x168U] = 0x3cdaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x169U] = 0x3cd7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x16aU] = 0x3cd4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x16bU] = 0x3cd0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x16cU] = 0x3ccdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x16dU] = 0x3ccaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x16eU] = 0x3cc7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x16fU] = 0x3cc4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x170U] = 0x3cc1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x171U] = 0x3cbeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x172U] = 0x3cbbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x173U] = 0x3cb8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x174U] = 0x3cb5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x175U] = 0x3cb2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x176U] = 0x3cafU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x177U] = 0x3cadU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x178U] = 0x3caaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x179U] = 0x3ca7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x17aU] = 0x3ca5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x17bU] = 0x3ca2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x17cU] = 0x3ca0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x17dU] = 0x3c9dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x17eU] = 0x3c9bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x17fU] = 0x3c98U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x180U] = 0x3c96U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x181U] = 0x3c91U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x182U] = 0x3c8dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x183U] = 0x3c89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x184U] = 0x3c84U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x185U] = 0x3c80U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x186U] = 0x3c79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x187U] = 0x3c71U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x188U] = 0x3c6aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x189U] = 0x3c63U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x18aU] = 0x3c5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x18bU] = 0x3c55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x18cU] = 0x3c4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x18dU] = 0x3c48U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x18eU] = 0x3c42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x18fU] = 0x3c3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x190U] = 0x3c36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x191U] = 0x3c30U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x192U] = 0x3c2bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x193U] = 0x3c26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x194U] = 0x3c21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x195U] = 0x3c1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x196U] = 0x3c17U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x197U] = 0x3c12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x198U] = 0x3c0eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x199U] = 0x3c09U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x19aU] = 0x3c05U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x19bU] = 0x3c01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x19cU] = 0x3bfaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x19dU] = 0x3bf2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x19eU] = 0x3bebU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x19fU] = 0x3be4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a0U] = 0x3bddU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a1U] = 0x3bd6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a2U] = 0x3bcfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a3U] = 0x3bc9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a4U] = 0x3bc3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a5U] = 0x3bbdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a6U] = 0x3bb7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a7U] = 0x3bb1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a8U] = 0x3bacU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1a9U] = 0x3ba7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1aaU] = 0x3ba2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1abU] = 0x3b9dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1acU] = 0x3b98U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1adU] = 0x3b93U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1aeU] = 0x3b8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1afU] = 0x3b8aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b0U] = 0x3b86U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b1U] = 0x3b82U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b2U] = 0x3b7cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b3U] = 0x3b74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b4U] = 0x3b6cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b5U] = 0x3b65U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b6U] = 0x3b5eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b7U] = 0x3b57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b8U] = 0x3b51U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1b9U] = 0x3b4aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1baU] = 0x3b44U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1bbU] = 0x3b3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1bcU] = 0x3b38U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1bdU] = 0x3b32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1beU] = 0x3b2dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1bfU] = 0x3b28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c0U] = 0x3b22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c1U] = 0x3b1dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c2U] = 0x3b19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c3U] = 0x3b14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c4U] = 0x3b0fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c5U] = 0x3b0bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c6U] = 0x3b07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c7U] = 0x3b03U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c8U] = 0x3afdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1c9U] = 0x3af5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1caU] = 0x3aeeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1cbU] = 0x3ae6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1ccU] = 0x3adfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1cdU] = 0x3ad8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1ceU] = 0x3ad2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1cfU] = 0x3acbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d0U] = 0x3ac5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d1U] = 0x3abfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d2U] = 0x3ab9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d3U] = 0x3ab3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d4U] = 0x3aaeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d5U] = 0x3aa9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d6U] = 0x3aa3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d7U] = 0x3a9eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d8U] = 0x3a99U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1d9U] = 0x3a95U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1daU] = 0x3a90U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1dbU] = 0x3a8cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1dcU] = 0x3a87U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1ddU] = 0x3a83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1deU] = 0x3a7eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1dfU] = 0x3a77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e0U] = 0x3a6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e1U] = 0x3a68U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e2U] = 0x3a61U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e3U] = 0x3a5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e4U] = 0x3a53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e5U] = 0x3a4cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e6U] = 0x3a46U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e7U] = 0x3a40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e8U] = 0x3a3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1e9U] = 0x3a34U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1eaU] = 0x3a2fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1ebU] = 0x3a2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1ecU] = 0x3a24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1edU] = 0x3a1fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1eeU] = 0x3a1aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1efU] = 0x3a16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f0U] = 0x3a11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f1U] = 0x3a0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f2U] = 0x3a08U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f3U] = 0x3a04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f4U] = 0x3a00U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f5U] = 0x39f8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f6U] = 0x39f0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f7U] = 0x39e9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f8U] = 0x39e2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1f9U] = 0x39dbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1faU] = 0x39d4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1fbU] = 0x39ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1fcU] = 0x39c7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1fdU] = 0x39c1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1feU] = 0x39bbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x1ffU] = 0x39b5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x200U] = 0x39b0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x201U] = 0x39a5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x202U] = 0x399bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x203U] = 0x3992U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x204U] = 0x3989U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x205U] = 0x3981U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x206U] = 0x3972U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x207U] = 0x3963U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x208U] = 0x3955U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x209U] = 0x3948U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x20aU] = 0x393cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x20bU] = 0x3931U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x20cU] = 0x3926U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x20dU] = 0x391cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x20eU] = 0x3913U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x20fU] = 0x390aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x210U] = 0x3901U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x211U] = 0x38f3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x212U] = 0x38e4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x213U] = 0x38d7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x214U] = 0x38caU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x215U] = 0x38bdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x216U] = 0x38b2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x217U] = 0x38a7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x218U] = 0x389dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x219U] = 0x3893U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x21aU] = 0x388bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x21bU] = 0x3882U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x21cU] = 0x3875U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x21dU] = 0x3866U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x21eU] = 0x3858U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x21fU] = 0x384bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x220U] = 0x383eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x221U] = 0x3833U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x222U] = 0x3828U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x223U] = 0x381eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x224U] = 0x3814U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x225U] = 0x380bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x226U] = 0x3803U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x227U] = 0x37f6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x228U] = 0x37e7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x229U] = 0x37d9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x22aU] = 0x37ccU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x22bU] = 0x37bfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x22cU] = 0x37b4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x22dU] = 0x37a9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x22eU] = 0x379fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x22fU] = 0x3795U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x230U] = 0x378cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x231U] = 0x3784U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x232U] = 0x3777U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x233U] = 0x3768U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x234U] = 0x375aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x235U] = 0x374dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x236U] = 0x3741U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x237U] = 0x3735U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x238U] = 0x372aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x239U] = 0x3720U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x23aU] = 0x3716U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x23bU] = 0x370dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x23cU] = 0x3704U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x23dU] = 0x36f9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x23eU] = 0x36eaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x23fU] = 0x36dbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x240U] = 0x36ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x241U] = 0x36c2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x242U] = 0x36b6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x243U] = 0x36abU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x244U] = 0x36a1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x245U] = 0x3697U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x246U] = 0x368eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x247U] = 0x3685U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x248U] = 0x367aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x249U] = 0x366bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x24aU] = 0x365dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x24bU] = 0x364fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x24cU] = 0x3643U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x24dU] = 0x3637U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x24eU] = 0x362cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x24fU] = 0x3621U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x250U] = 0x3618U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x251U] = 0x360eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x252U] = 0x3606U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x253U] = 0x35fcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x254U] = 0x35ecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x255U] = 0x35deU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x256U] = 0x35d1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x257U] = 0x35c4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x258U] = 0x35b8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x259U] = 0x35adU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x25aU] = 0x35a2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x25bU] = 0x3599U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x25cU] = 0x358fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x25dU] = 0x3587U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x25eU] = 0x357dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x25fU] = 0x356eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x260U] = 0x355fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x261U] = 0x3552U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x262U] = 0x3545U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x263U] = 0x3539U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x264U] = 0x352eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x265U] = 0x3523U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x266U] = 0x3519U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x267U] = 0x3510U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x268U] = 0x3507U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x269U] = 0x34feU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x26aU] = 0x34efU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x26bU] = 0x34e0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x26cU] = 0x34d3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x26dU] = 0x34c6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x26eU] = 0x34baU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x26fU] = 0x34afU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x270U] = 0x34a4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x271U] = 0x349aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x272U] = 0x3491U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x273U] = 0x3488U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x274U] = 0x3480U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x275U] = 0x3470U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x276U] = 0x3462U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x277U] = 0x3454U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x278U] = 0x3447U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x279U] = 0x343bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x27aU] = 0x3430U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x27bU] = 0x3425U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x27cU] = 0x341bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x27dU] = 0x3412U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x27eU] = 0x3409U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x27fU] = 0x3401U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x280U] = 0x33f2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x281U] = 0x33d5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x282U] = 0x33bcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x283U] = 0x33a6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x284U] = 0x3393U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x285U] = 0x3381U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x286U] = 0x3364U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x287U] = 0x3349U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x288U] = 0x3332U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x289U] = 0x331dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x28aU] = 0x330aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x28bU] = 0x32f4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x28cU] = 0x32d8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x28dU] = 0x32beU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x28eU] = 0x32a8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x28fU] = 0x3294U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x290U] = 0x3283U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x291U] = 0x3267U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x292U] = 0x324cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x293U] = 0x3234U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x294U] = 0x321fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x295U] = 0x320cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x296U] = 0x31f7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x297U] = 0x31daU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x298U] = 0x31c1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x299U] = 0x31aaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x29aU] = 0x3196U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x29bU] = 0x3184U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x29cU] = 0x316aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x29dU] = 0x314eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x29eU] = 0x3136U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x29fU] = 0x3121U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a0U] = 0x310eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a1U] = 0x30faU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a2U] = 0x30ddU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a3U] = 0x30c3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a4U] = 0x30acU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a5U] = 0x3098U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a6U] = 0x3086U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a7U] = 0x306cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a8U] = 0x3050U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2a9U] = 0x3038U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2aaU] = 0x3022U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2abU] = 0x300fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2acU] = 0x2ffdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2adU] = 0x2fdfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2aeU] = 0x2fc5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2afU] = 0x2faeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b0U] = 0x2f99U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b1U] = 0x2f87U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b2U] = 0x2f6fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b3U] = 0x2f53U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b4U] = 0x2f3aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b5U] = 0x2f24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b6U] = 0x2f11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b7U] = 0x2f00U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b8U] = 0x2ee2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2b9U] = 0x2ec7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2baU] = 0x2eb0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2bbU] = 0x2e9bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2bcU] = 0x2e89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2bdU] = 0x2e72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2beU] = 0x2e55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2bfU] = 0x2e3cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c0U] = 0x2e26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c1U] = 0x2e13U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c2U] = 0x2e01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c3U] = 0x2de4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c4U] = 0x2dc9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c5U] = 0x2db2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c6U] = 0x2d9dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c7U] = 0x2d8aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c8U] = 0x2d74U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2c9U] = 0x2d58U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2caU] = 0x2d3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2cbU] = 0x2d28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2ccU] = 0x2d14U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2cdU] = 0x2d03U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2ceU] = 0x2ce7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2cfU] = 0x2cccU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d0U] = 0x2cb4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d1U] = 0x2c9fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d2U] = 0x2c8cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d3U] = 0x2c77U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d4U] = 0x2c5aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d5U] = 0x2c40U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d6U] = 0x2c2aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d7U] = 0x2c16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d8U] = 0x2c04U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2d9U] = 0x2be9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2daU] = 0x2bceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2dbU] = 0x2bb6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2dcU] = 0x2ba0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2ddU] = 0x2b8eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2deU] = 0x2b7aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2dfU] = 0x2b5dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e0U] = 0x2b43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e1U] = 0x2b2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e2U] = 0x2b18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e3U] = 0x2b06U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e4U] = 0x2aecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e5U] = 0x2ad0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e6U] = 0x2ab8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e7U] = 0x2aa2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e8U] = 0x2a8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2e9U] = 0x2a7dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2eaU] = 0x2a5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2ebU] = 0x2a45U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2ecU] = 0x2a2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2edU] = 0x2a19U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2eeU] = 0x2a07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2efU] = 0x29efU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f0U] = 0x29d3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f1U] = 0x29baU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f2U] = 0x29a4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f3U] = 0x2991U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f4U] = 0x2980U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f5U] = 0x2962U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f6U] = 0x2947U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f7U] = 0x2930U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f8U] = 0x291bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2f9U] = 0x2909U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2faU] = 0x28f1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2fbU] = 0x28d5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2fcU] = 0x28bcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2fdU] = 0x28a6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2feU] = 0x2892U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x2ffU] = 0x2881U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x300U] = 0x2864U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x301U] = 0x2832U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x302U] = 0x280aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x303U] = 0x27d8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x304U] = 0x27a8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x305U] = 0x2783U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x306U] = 0x274cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x307U] = 0x271fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x308U] = 0x26f7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x309U] = 0x26c0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x30aU] = 0x2696U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x30bU] = 0x2669U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x30cU] = 0x2636U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x30dU] = 0x260eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x30eU] = 0x25dcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x30fU] = 0x25acU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x310U] = 0x2586U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x311U] = 0x2550U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x312U] = 0x2522U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x313U] = 0x24fdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x314U] = 0x24c5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x315U] = 0x2499U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x316U] = 0x246fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x317U] = 0x243aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x318U] = 0x2411U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x319U] = 0x23e1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x31aU] = 0x23b0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x31bU] = 0x2389U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x31cU] = 0x2355U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x31dU] = 0x2326U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x31eU] = 0x2301U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x31fU] = 0x22c9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x320U] = 0x229dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x321U] = 0x2274U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x322U] = 0x223eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x323U] = 0x2214U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x324U] = 0x21e7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x325U] = 0x21b4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x326U] = 0x218cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x327U] = 0x215aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x328U] = 0x212aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x329U] = 0x2104U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x32aU] = 0x20ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x32bU] = 0x20a0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x32cU] = 0x207aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x32dU] = 0x2042U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x32eU] = 0x2017U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x32fU] = 0x1fecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x330U] = 0x1fb8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x331U] = 0x1f8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x332U] = 0x1f5fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x333U] = 0x1f2eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x334U] = 0x1f07U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x335U] = 0x1ed3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x336U] = 0x1ea4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x337U] = 0x1e7fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x338U] = 0x1e47U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x339U] = 0x1e1bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x33aU] = 0x1df1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x33bU] = 0x1dbcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x33cU] = 0x1d92U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x33dU] = 0x1d64U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x33eU] = 0x1d32U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x33fU] = 0x1d0aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x340U] = 0x1cd7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x341U] = 0x1ca8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x342U] = 0x1c83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x343U] = 0x1c4bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x344U] = 0x1c1eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x345U] = 0x1bf7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x346U] = 0x1bc0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x347U] = 0x1b96U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x348U] = 0x1b69U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x349U] = 0x1b36U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x34aU] = 0x1b0dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x34bU] = 0x1adcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x34cU] = 0x1aacU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x34dU] = 0x1a86U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x34eU] = 0x1a50U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x34fU] = 0x1a22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x350U] = 0x19fcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x351U] = 0x19c5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x352U] = 0x1999U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x353U] = 0x196eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x354U] = 0x193aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x355U] = 0x1911U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x356U] = 0x18e1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x357U] = 0x18afU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x358U] = 0x1889U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x359U] = 0x1855U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x35aU] = 0x1826U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x35bU] = 0x1801U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x35cU] = 0x17c9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x35dU] = 0x179dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x35eU] = 0x1774U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x35fU] = 0x173eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x360U] = 0x1714U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x361U] = 0x16e6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x362U] = 0x16b3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x363U] = 0x168cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x364U] = 0x165aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x365U] = 0x162aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x366U] = 0x1604U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x367U] = 0x15ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x368U] = 0x15a0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x369U] = 0x157aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x36aU] = 0x1542U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x36bU] = 0x1517U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x36cU] = 0x14ecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x36dU] = 0x14b8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x36eU] = 0x148fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x36fU] = 0x145fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x370U] = 0x142dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x371U] = 0x1407U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x372U] = 0x13d2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x373U] = 0x13a4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x374U] = 0x137fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x375U] = 0x1347U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x376U] = 0x131bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x377U] = 0x12f1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x378U] = 0x12bcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x379U] = 0x1292U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x37aU] = 0x1264U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x37bU] = 0x1231U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x37cU] = 0x120aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x37dU] = 0x11d7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x37eU] = 0x11a8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x37fU] = 0x1183U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x380U] = 0x114bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x381U] = 0x10f7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x382U] = 0x1096U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x383U] = 0x1035U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x384U] = 0xfdcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x385U] = 0xf86U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x386U] = 0xf22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x387U] = 0xec4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x388U] = 0xe6eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x389U] = 0xe11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x38aU] = 0xdafU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x38bU] = 0xd55U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x38cU] = 0xd01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x38dU] = 0xc9cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x38eU] = 0xc3eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x38fU] = 0xbe6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x390U] = 0xb8cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x391U] = 0xb29U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x392U] = 0xaceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x393U] = 0xa79U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x394U] = 0xa17U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x395U] = 0x9b7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x396U] = 0x95fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x397U] = 0x907U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x398U] = 0x8a4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x399U] = 0x847U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x39aU] = 0x7f1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x39bU] = 0x792U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x39cU] = 0x731U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x39dU] = 0x6d7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x39eU] = 0x682U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x39fU] = 0x61eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a0U] = 0x5c0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a1U] = 0x569U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a2U] = 0x50dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a3U] = 0x4abU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a4U] = 0x450U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a5U] = 0x3fcU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a6U] = 0x399U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a7U] = 0x339U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a8U] = 0x2e1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3a9U] = 0x288U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3aaU] = 0x226U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3abU] = 0x1c9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3acU] = 0x174U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3adU] = 0x114U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3aeU] = 0xb3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3afU] = 0x6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b0U] = 0x42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b1U] = 0x28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b2U] = 0x18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b3U] = 0xfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b4U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b5U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b6U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b7U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b8U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[0x3b9U] = 1U;
    __Vilp = 0x3baU;
    while ((__Vilp <= 0x3ffU)) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[__Vilp] = 0U;
        __Vilp = ((IData)(1U) + __Vilp);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0U] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[1U] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[2U] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[3U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[4U] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[5U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[6U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[7U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[8U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[9U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xaU] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xbU] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xcU] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xdU] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xeU] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xfU] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x10U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x11U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x12U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x13U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x14U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x15U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x16U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x17U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x18U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x19U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x1aU] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x1bU] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x1cU] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x1dU] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x1eU] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x1fU] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x20U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x21U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x22U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x23U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x24U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x25U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x26U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x27U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x28U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x29U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x2aU] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x2bU] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x2cU] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x2dU] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x2eU] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x2fU] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x30U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x31U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x32U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x33U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x34U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x35U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x36U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x37U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x38U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x39U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x3aU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x3bU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x3cU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x3dU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x3eU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x3fU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x40U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x41U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x42U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x43U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x44U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x45U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x46U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x47U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x48U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x49U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x4aU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x4bU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x4cU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x4dU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x4eU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x4fU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x50U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x51U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x52U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x53U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x54U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x55U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x56U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x57U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x58U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x59U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x5aU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x5bU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x5cU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x5dU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x5eU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x5fU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x60U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x61U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x62U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x63U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x64U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x65U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x66U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x67U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x68U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x69U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x6aU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x6bU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x6cU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x6dU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x6eU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x6fU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x70U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x71U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x72U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x73U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x74U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x75U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x76U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x77U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x78U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x79U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x7aU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x7bU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x7cU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x7dU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x7eU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x7fU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x80U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x81U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x82U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x83U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x84U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x85U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x86U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x87U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x88U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x89U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x8aU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x8bU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x8cU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x8dU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x8eU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x8fU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x90U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x91U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x92U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x93U] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x94U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x95U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x96U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x97U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x98U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x99U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x9aU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x9bU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x9cU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x9dU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x9eU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0x9fU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa0U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa1U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa2U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa3U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa4U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa5U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa6U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa7U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa8U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xa9U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xaaU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xabU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xacU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xadU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xaeU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xafU] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb0U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb1U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb2U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb3U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb4U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb5U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb6U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb7U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb8U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xb9U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xbaU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xbbU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xbcU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xbdU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xbeU] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xbfU] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc0U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc1U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc2U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc3U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc4U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc5U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc6U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc7U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc8U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xc9U] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xcaU] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xcbU] = 4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xccU] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xcdU] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xceU] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xcfU] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd0U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd1U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd2U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd3U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd4U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd5U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd6U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd7U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd8U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xd9U] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xdaU] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xdbU] = 6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xdcU] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xddU] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xdeU] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xdfU] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe0U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe1U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe2U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe3U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe4U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe5U] = 7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe6U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe7U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe8U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xe9U] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xeaU] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xebU] = 8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xecU] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xedU] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xeeU] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xefU] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf0U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf1U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf2U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf3U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf4U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf5U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf6U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf7U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf8U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xf9U] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xfaU] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xfbU] = 0xaU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xfcU] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xfdU] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xfeU] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[0xffU] = 0xbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[0U] = 0x6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[1U] = 0x42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[2U] = 0x28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[3U] = 0x18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[4U] = 0xfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[5U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[6U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[7U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[8U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[9U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[0xaU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0U] = 0x15b3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[1U] = 0x14d1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[2U] = 0x13eeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[3U] = 0x130bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[4U] = 0x1279U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[5U] = 0x11ebU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[6U] = 0x1108U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[7U] = 0x1021U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[8U] = 0xf43U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[9U] = 0xed8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0xaU] = 0xdedU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0xbU] = 0xd8fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0xcU] = 0xd29U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0xdU] = 0xc26U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0xeU] = 0xb5cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0xfU] = 0xb22U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x10U] = 0xa6bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x11U] = 0x9d9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x12U] = 0x936U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x13U] = 0x8b4U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x14U] = 0x87aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x15U] = 0x814U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x16U] = 0x731U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x17U] = 0x6d7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x18U] = 0x689U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x19U] = 0x5f7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x1aU] = 0x5e1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x1bU] = 0x52aU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x1cU] = 0x4bfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x1dU] = 0x482U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x1eU] = 0x427U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x1fU] = 0x446U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x20U] = 0x38fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x21U] = 0x331U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x22U] = 0x2f7U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x23U] = 0x2e1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x24U] = 0x2bbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x25U] = 0x291U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x26U] = 0x1ffU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x27U] = 0x1e9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x28U] = 0x183U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x29U] = 0x168U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x2aU] = 0x1bfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x2bU] = 0xf1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x2cU] = 0xdbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x2dU] = 0xc5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x2eU] = 0xafU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x2fU] = 0x78U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x30U] = 0x83U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x31U] = 0x6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x32U] = 0x57U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x33U] = 0x21U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x34U] = 0x1cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x35U] = 0x16U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x36U] = 0x80U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x37U] = 0x11U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x38U] = 0x2cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x39U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x3aU] = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x3bU] = 0x67U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x3cU] = 0x3dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x3dU] = 0x3bU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x3eU] = 0x25U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x3fU] = 0x33U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x40U] = 0x4eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x41U] = 0x60U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x42U] = 0x81U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x43U] = 0xb0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x44U] = 0xc2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x45U] = 0xedU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x46U] = 0x17cU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x47U] = 0x152U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x48U] = 0x160U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x49U] = 0x16eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x4aU] = 0x1e9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x4bU] = 0x1d2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x4cU] = 0x201U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x4dU] = 0x280U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x4eU] = 0x27eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x4fU] = 0x2b8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x50U] = 0x2ceU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x51U] = 0x349U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x52U] = 0x36fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x53U] = 0x3aeU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x54U] = 0x3e0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x55U] = 0x412U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x56U] = 0x4a1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x57U] = 0x4d3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x58U] = 0x502U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x59U] = 0x538U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x5aU] = 0x57eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x5bU] = 0x5ddU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x5cU] = 0x61fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x5dU] = 0x686U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x5eU] = 0x6ecU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x5fU] = 0x777U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x60U] = 0x785U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x61U] = 0x7dbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x62U] = 0x840U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x63U] = 0x895U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x64U] = 0x8fbU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x65U] = 0x962U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x66U] = 0x9c2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x67U] = 0xa51U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x68U] = 0xab1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x69U] = 0xaf8U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x6aU] = 0xb72U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x6bU] = 0xbd3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x6cU] = 0xc62U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x6dU] = 0xcc2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x6eU] = 0xd24U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x6fU] = 0xdabU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x70U] = 0xe12U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x71U] = 0xe89U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x72U] = 0xf01U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x73U] = 0xf90U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x74U] = 0x1007U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x75U] = 0x107fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x76U] = 0x10f6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x77U] = 0x1185U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x78U] = 0x11fdU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x79U] = 0x1280U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x7aU] = 0x1303U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x7bU] = 0x138fU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x7cU] = 0x1415U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x7dU] = 0x149eU;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x7eU] = 0x1529U;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[0x7fU] = 0x15b6U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[0U] = 0x6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[1U] = 0x42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[2U] = 0x28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[3U] = 0x18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[4U] = 0xfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[5U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[6U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[7U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[8U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[9U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[0xaU] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[0U] = 0x6dU;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[1U] = 0x42U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[2U] = 0x28U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[3U] = 0x18U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[4U] = 0xfU;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[5U] = 9U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[6U] = 5U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[7U] = 3U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[8U] = 2U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[9U] = 1U;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[0xaU] = 1U;
}

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___eval_final(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___dump_triggers__stl(Vbf16_expe_equiv_top___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vbf16_expe_equiv_top___024root___eval_phase__stl(Vbf16_expe_equiv_top___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___eval_settle(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_settle\n"); );
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
            Vbf16_expe_equiv_top___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("tests/bf16_expe_equiv_top.sv", 26, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vbf16_expe_equiv_top___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___dump_triggers__stl(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

extern const VlUnpacked<SData/*15:0*/, 1024> Vbf16_expe_equiv_top__ConstPool__TABLE_hf7ff168e_0;

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___stl_sequent__TOP__0(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___stl_sequent__TOP__0\n"); );
    // Init
    QData/*58:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__ax_unsigned;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__ax_unsigned = 0;
    QData/*61:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__neg_ax;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__neg_ax = 0;
    QData/*61:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__b_aligned;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__b_aligned = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shift_amt;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shift_amt = 0;
    QData/*61:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shifted_res;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shifted_res = 0;
    IData/*31:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0;
    CData/*0:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__lsb_bit;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__lsb_bit = 0;
    CData/*0:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__guard_bit;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__guard_bit = 0;
    CData/*0:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__round_up;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__round_up = 0;
    QData/*58:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sticky_masked;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sticky_masked = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__adjusted_exp;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__adjusted_exp = 0;
    CData/*7:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0;
    CData/*6:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0;
    QData/*58:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__ax_unsigned;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__ax_unsigned = 0;
    QData/*61:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__neg_ax;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__neg_ax = 0;
    QData/*61:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__b_aligned;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__b_aligned = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shift_amt;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shift_amt = 0;
    QData/*61:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shifted_res;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shifted_res = 0;
    IData/*31:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0;
    CData/*0:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__lsb_bit;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__lsb_bit = 0;
    CData/*0:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__guard_bit;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__guard_bit = 0;
    CData/*0:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__round_up;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__round_up = 0;
    QData/*58:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sticky_masked;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sticky_masked = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__adjusted_exp;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__adjusted_exp = 0;
    CData/*7:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0;
    CData/*6:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0;
    SData/*9:0*/ __Vtableidx1;
    __Vtableidx1 = 0;
    // Body
    vlSelf->o_exp2_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__bf16_out;
    vlSelf->o_exp2opt_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__bf16_out;
    vlSelf->o_lut_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [7U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [1U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [2U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [3U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [4U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [5U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [6U];
    vlSelf->o_hybrid_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [7U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [1U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [2U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [3U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [4U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [5U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [6U];
    vlSelf->o_cut_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [4U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [1U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [2U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [3U];
    vlSelf->o_poly4_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__core_data;
    vlSelf->o_poly4dsp_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__core_data;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_src 
        = ((0x80U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                     << 3U)) | (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                                         >> 5U)));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_src 
        = ((0x80U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
                     << 3U)) | (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
                                         >> 5U)));
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_eo_code;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_route;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_eo;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_int_part;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_frac;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_route;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_eo;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_int_part;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_frac;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT____VdfgTmp_h0f01a197__0 
        = (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_prod 
           >> (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_shift));
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__base_count 
        = (0x1ffU & ((IData)(0x1f4U) + (((IData)(0x80U) 
                                         - (0x7fU & 
                                            ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_frac) 
                                             >> 9U))) 
                                        + (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_cand_dev))));
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__ladder_count 
        = (0x1ffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_base_count) 
                     + (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_below_cut)));
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____VdfgTmp_h785166bd__0 
        = (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_prod 
           >> (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_shift));
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail[0U] 
        = ((0xaU >= (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_tail_addr))
            ? vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom
           [vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_tail_addr]
            : 0U);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__scaled_comb 
        = (0x1ffffffffffULL & ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_t7)) 
                               * (QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_onehot))));
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[0U] 
        = ((0xaU >= (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_tail_addr))
            ? vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom
           [vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_tail_addr]
            : 0U);
    __Vtableidx1 = (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_mantissa) 
                     << 3U) | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_sparse_exp_index));
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_sparse__DOT__data_comb 
        = Vbf16_expe_equiv_top__ConstPool__TABLE_hf7ff168e_0
        [__Vtableidx1];
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[0U] 
        = (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                     >> 0xcU));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay[0U] 
        = (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
                     >> 0xcU));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shift_amt 
        = (0x1ffU & ((IData)(0x3dU) - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_s)));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shifted_res 
        = (0x3fffffffffffffffULL & VL_SHIFTL_QQI(62,62,9, vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__res_s, (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shift_amt)));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__poly_mant_comb 
        = (0x7ffffffffffffffULL & (bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shifted_res 
                                   >> 3U));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shift_amt 
        = (0x1ffU & ((IData)(0x3dU) - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_s)));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shifted_res 
        = (0x3fffffffffffffffULL & VL_SHIFTL_QQI(62,62,9, vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__res_s, (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shift_amt)));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__poly_mant_comb 
        = (0x7ffffffffffffffULL & (bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shifted_res 
                                   >> 3U));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_m_ext;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__adjusted_exp 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_base_exp;
    if ((0x100U & (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext))) {
        bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__adjusted_exp 
            = (0x1ffU & ((IData)(1U) + (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__adjusted_exp)));
        bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext 
            = (0x1ffU & VL_SHIFTR_III(9,9,32, (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext), 1U));
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb 
        = (0x1fffffU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb);
    if ((0U == (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext))) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb 
            = (1U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb);
    } else if (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_is_sub) 
                & (~ ((IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext) 
                      >> 7U)))) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb 
            = ((0x20000fU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb) 
               | (0x181000U | (0xfe0U & ((IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext) 
                                         << 5U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb 
            = (2U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb);
    } else {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb 
            = ((0x20000fU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb) 
               | (0x10U | (((IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__adjusted_exp) 
                            << 0xcU) | (0xfe0U & ((IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__result_m_ext) 
                                                  << 5U)))));
    }
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_m_ext;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__adjusted_exp 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_base_exp;
    if ((0x100U & (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext))) {
        bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__adjusted_exp 
            = (0x1ffU & ((IData)(1U) + (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__adjusted_exp)));
        bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext 
            = (0x1ffU & VL_SHIFTR_III(9,9,32, (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext), 1U));
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb = 0U;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb 
        = (0x1fffffU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb);
    if ((0U == (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext))) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb 
            = (1U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb);
    } else if (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_is_sub) 
                & (~ ((IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext) 
                      >> 7U)))) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb 
            = ((0x20000fU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb) 
               | (0x181000U | (0xfe0U & ((IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext) 
                                         << 5U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb 
            = (2U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb);
    } else {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb 
            = ((0x20000fU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb) 
               | (0x10U | (((IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__adjusted_exp) 
                            << 0xcU) | (0xfe0U & ((IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__result_m_ext) 
                                                  << 5U)))));
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_int_part;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s4_int_part;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__ax_unsigned 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__ax_core;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__neg_ax 
        = (0x3fffffffffffffffULL & (- VL_EXTENDS_QQ(62,60, bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__ax_unsigned)));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__b_aligned 
        = (0x3fffffffffffffffULL & ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__coeff_b_stage)) 
                                    << 0x26U));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__calc_res 
        = (0x3fffffffffffffffULL & (bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__b_aligned 
                                    + bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__neg_ax));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__ax_unsigned 
        = (0x7ffffffffffffffULL & VL_SHIFTL_QQI(59,59,32, vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__ax_core, 8U));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__neg_ax 
        = (0x3fffffffffffffffULL & (- VL_EXTENDS_QQ(62,60, bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__ax_unsigned)));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__b_aligned 
        = (0x3fffffffffffffffULL & ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__coeff_b_stage)) 
                                    << 0x26U));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__calc_res 
        = (0x3fffffffffffffffULL & (bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__b_aligned 
                                    + bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__neg_ax));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sticky_masked 
        = (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_mant 
           & (((QData)((IData)(VL_LTS_III(9, 0x3bU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
               << 0x3aU) | (((QData)((IData)(VL_LTS_III(9, 0x3aU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                             << 0x39U) | (((QData)((IData)(
                                                           VL_LTS_III(9, 0x39U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                           << 0x38U) 
                                          | (((QData)((IData)(
                                                              VL_LTS_III(9, 0x38U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                              << 0x37U) 
                                             | (((QData)((IData)(
                                                                 VL_LTS_III(9, 0x37U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                 << 0x36U) 
                                                | (((QData)((IData)(
                                                                    VL_LTS_III(9, 0x36U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                    << 0x35U) 
                                                   | (((QData)((IData)(
                                                                       VL_LTS_III(9, 0x35U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                       << 0x34U) 
                                                      | (((QData)((IData)(
                                                                          VL_LTS_III(9, 0x34U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                          << 0x33U) 
                                                         | (((QData)((IData)(
                                                                             VL_LTS_III(9, 0x33U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                             << 0x32U) 
                                                            | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x32U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                << 0x31U) 
                                                               | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x31U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                   << 0x30U) 
                                                                  | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x30U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                      << 0x2fU) 
                                                                     | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2fU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                         << 0x2eU) 
                                                                        | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2eU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                            << 0x2dU) 
                                                                           | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2dU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                               << 0x2cU) 
                                                                              | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2cU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2bU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2bU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2aU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2aU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x29U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x29U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x28U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x28U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x27U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x27U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x26U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x26U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x25U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x25U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x24U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x24U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x23U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x23U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x22U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x22U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x21U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x21U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x20U) 
                                                                                | (QData)((IData)(
                                                                                ((VL_LTS_III(9, 0x20U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1fU) 
                                                                                | ((VL_LTS_III(9, 0x1fU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1eU) 
                                                                                | ((VL_LTS_III(9, 0x1eU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1dU) 
                                                                                | ((VL_LTS_III(9, 0x1dU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1cU) 
                                                                                | ((VL_LTS_III(9, 0x1cU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1bU) 
                                                                                | ((VL_LTS_III(9, 0x1bU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1aU) 
                                                                                | ((VL_LTS_III(9, 0x1aU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x19U) 
                                                                                | ((VL_LTS_III(9, 0x19U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x18U) 
                                                                                | ((VL_LTS_III(9, 0x18U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x17U) 
                                                                                | ((VL_LTS_III(9, 0x17U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x16U) 
                                                                                | ((VL_LTS_III(9, 0x16U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x15U) 
                                                                                | ((VL_LTS_III(9, 0x15U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x14U) 
                                                                                | ((VL_LTS_III(9, 0x14U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x13U) 
                                                                                | ((VL_LTS_III(9, 0x13U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x12U) 
                                                                                | ((VL_LTS_III(9, 0x12U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x11U) 
                                                                                | ((VL_LTS_III(9, 0x11U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x10U) 
                                                                                | ((VL_LTS_III(9, 0x10U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xfU) 
                                                                                | ((VL_LTS_III(9, 0xfU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xeU) 
                                                                                | ((VL_LTS_III(9, 0xeU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xdU) 
                                                                                | ((VL_LTS_III(9, 0xdU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xcU) 
                                                                                | ((VL_LTS_III(9, 0xcU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xbU) 
                                                                                | ((VL_LTS_III(9, 0xbU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xaU) 
                                                                                | ((VL_LTS_III(9, 0xaU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 9U) 
                                                                                | ((VL_LTS_III(9, 9U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 8U) 
                                                                                | ((VL_LTS_III(9, 8U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 7U) 
                                                                                | ((VL_LTS_III(9, 7U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 6U) 
                                                                                | ((VL_LTS_III(9, 6U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 5U) 
                                                                                | ((VL_LTS_III(9, 5U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 4U) 
                                                                                | ((VL_LTS_III(9, 4U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 3U) 
                                                                                | ((VL_LTS_III(9, 3U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 2U) 
                                                                                | ((VL_LTS_III(9, 2U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                                                                << 1U) 
                                                                                | VL_LTS_III(9, 1U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sticky_masked 
        = (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_mant 
           & (((QData)((IData)(VL_LTS_III(9, 0x3bU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
               << 0x3aU) | (((QData)((IData)(VL_LTS_III(9, 0x3aU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                             << 0x39U) | (((QData)((IData)(
                                                           VL_LTS_III(9, 0x39U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                           << 0x38U) 
                                          | (((QData)((IData)(
                                                              VL_LTS_III(9, 0x38U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                              << 0x37U) 
                                             | (((QData)((IData)(
                                                                 VL_LTS_III(9, 0x37U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                 << 0x36U) 
                                                | (((QData)((IData)(
                                                                    VL_LTS_III(9, 0x36U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                    << 0x35U) 
                                                   | (((QData)((IData)(
                                                                       VL_LTS_III(9, 0x35U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                       << 0x34U) 
                                                      | (((QData)((IData)(
                                                                          VL_LTS_III(9, 0x34U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                          << 0x33U) 
                                                         | (((QData)((IData)(
                                                                             VL_LTS_III(9, 0x33U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                             << 0x32U) 
                                                            | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x32U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                << 0x31U) 
                                                               | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x31U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                   << 0x30U) 
                                                                  | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x30U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                      << 0x2fU) 
                                                                     | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2fU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                         << 0x2eU) 
                                                                        | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2eU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                            << 0x2dU) 
                                                                           | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2dU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                               << 0x2cU) 
                                                                              | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2cU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2bU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2bU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x2aU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2aU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x29U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x29U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x28U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x28U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x27U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x27U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x26U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x26U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x25U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x25U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x24U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x24U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x23U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x23U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x22U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x22U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x21U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x21U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
                                                                                << 0x20U) 
                                                                                | (QData)((IData)(
                                                                                ((VL_LTS_III(9, 0x20U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1fU) 
                                                                                | ((VL_LTS_III(9, 0x1fU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1eU) 
                                                                                | ((VL_LTS_III(9, 0x1eU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1dU) 
                                                                                | ((VL_LTS_III(9, 0x1dU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1cU) 
                                                                                | ((VL_LTS_III(9, 0x1cU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1bU) 
                                                                                | ((VL_LTS_III(9, 0x1bU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x1aU) 
                                                                                | ((VL_LTS_III(9, 0x1aU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x19U) 
                                                                                | ((VL_LTS_III(9, 0x19U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x18U) 
                                                                                | ((VL_LTS_III(9, 0x18U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x17U) 
                                                                                | ((VL_LTS_III(9, 0x17U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x16U) 
                                                                                | ((VL_LTS_III(9, 0x16U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x15U) 
                                                                                | ((VL_LTS_III(9, 0x15U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x14U) 
                                                                                | ((VL_LTS_III(9, 0x14U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x13U) 
                                                                                | ((VL_LTS_III(9, 0x13U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x12U) 
                                                                                | ((VL_LTS_III(9, 0x12U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x11U) 
                                                                                | ((VL_LTS_III(9, 0x11U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0x10U) 
                                                                                | ((VL_LTS_III(9, 0x10U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xfU) 
                                                                                | ((VL_LTS_III(9, 0xfU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xeU) 
                                                                                | ((VL_LTS_III(9, 0xeU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xdU) 
                                                                                | ((VL_LTS_III(9, 0xdU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xcU) 
                                                                                | ((VL_LTS_III(9, 0xcU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xbU) 
                                                                                | ((VL_LTS_III(9, 0xbU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 0xaU) 
                                                                                | ((VL_LTS_III(9, 0xaU, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 9U) 
                                                                                | ((VL_LTS_III(9, 9U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 8U) 
                                                                                | ((VL_LTS_III(9, 8U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 7U) 
                                                                                | ((VL_LTS_III(9, 7U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 6U) 
                                                                                | ((VL_LTS_III(9, 6U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 5U) 
                                                                                | ((VL_LTS_III(9, 5U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 4U) 
                                                                                | ((VL_LTS_III(9, 4U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 3U) 
                                                                                | ((VL_LTS_III(9, 3U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 2U) 
                                                                                | ((VL_LTS_III(9, 2U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                                                                << 1U) 
                                                                                | VL_LTS_III(9, 1U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));
    vlSelf->o_exp2_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__gen_axi_pipelined__DOT__vld_sr) 
                                   >> 0xbU));
    vlSelf->o_exp2opt_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__gen_axi_pipelined__DOT__vld_sr) 
                                      >> 0xbU));
    vlSelf->o_lut_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__gen_axi_pipelined__DOT__vld_sr) 
                                  >> 0xbU));
    vlSelf->o_hybrid_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                     >> 0xbU));
    vlSelf->o_cut_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                  >> 0xbU));
    vlSelf->o_poly4_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                    >> 0xbU));
    vlSelf->o_poly4dsp_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                       >> 0xbU));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s2_eo_code;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s2_eo_code;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb 
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
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_unified_shift__DOT__unified_shifted 
        = (0x7fffffffffffULL & (VL_LTES_III(32, 0U, 
                                            VL_EXTENDS_II(32,9, 
                                                          vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                                                          [2U]))
                                 ? VL_SHIFTL_QQI(47,47,9, 
                                                 ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_out)) 
                                                  << 9U), 
                                                 vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                                                 [2U])
                                 : VL_SHIFTR_QQI(47,47,9, 
                                                 ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_out)) 
                                                  << 9U), 
                                                 (0x1ffU 
                                                  & (- 
                                                     vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                                                     [2U])))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__unified_shifted 
        = (0x7fffffffffffULL & (VL_SHIFTL_QQI(47,47,32, (QData)((IData)(
                                                                        ((0x7fffU 
                                                                          & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_out 
                                                                             >> 0x10U)) 
                                                                         * 
                                                                         (0x1ffffU 
                                                                          & ((IData)(1U) 
                                                                             << 
                                                                             (0x1fU 
                                                                              & ((IData)(9U) 
                                                                                + 
                                                                                vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay
                                                                                [2U]))))))), 0x10U) 
                                | (0x1ffffffffULL & 
                                   ((QData)((IData)(
                                                    (0xffffU 
                                                     & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_out))) 
                                    * (QData)((IData)(
                                                      (0x1ffffU 
                                                       & ((IData)(1U) 
                                                          << 
                                                          (0x1fU 
                                                           & ((IData)(9U) 
                                                              + 
                                                              vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay
                                                              [2U]))))))))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__final_exponent 
        = (0x1ffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_poly_exp) 
                     + vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay
                     [4U]));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__is_sub 
        = VL_GTS_III(9, 0x182U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__final_exponent));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__shift_9 
        = ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__is_sub)
            ? (0x1ffU & ((IData)(0x33U) + ((IData)(0x182U) 
                                           - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__final_exponent))))
            : 0x33U);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__final_exponent 
        = (0x1ffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s6_poly_exp) 
                     + vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay
                     [4U]));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__is_sub 
        = VL_GTS_III(9, 0x182U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__final_exponent));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__shift_9 
        = ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__is_sub)
            ? (0x1ffU & ((IData)(0x33U) + ((IData)(0x182U) 
                                           - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__final_exponent))))
            : 0x33U);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_comb = 0x1ffU;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0x3dU;
    {
        while (VL_LTES_III(32, 0U, bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) {
            if (((0x3dU >= (0x3fU & bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) 
                 && (1U & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__calc_res 
                                   >> (0x3fU & bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)))))) {
                vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_comb 
                    = (0x1ffU & bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i);
                goto __Vlabel1;
            }
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                = (bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                   - (IData)(1U));
        }
        __Vlabel1: ;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_comb = 0x1ffU;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0x3dU;
    {
        while (VL_LTES_III(32, 0U, bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) {
            if (((0x3dU >= (0x3fU & bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) 
                 && (1U & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__calc_res 
                                   >> (0x3fU & bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i)))))) {
                vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_comb 
                    = (0x1ffU & bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i);
                goto __Vlabel2;
            }
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                = (bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                   - (IData)(1U));
        }
        __Vlabel2: ;
    }
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__lsb_bit 
        = (VL_GTS_III(32, 0x3bU, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift))) 
           & ((0x3aU >= (0x3fU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift))) 
              && (1U & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_mant 
                                >> (0x3fU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))))));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__guard_bit 
        = ((VL_LTS_III(32, 0U, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift))) 
            & VL_GTES_III(32, 0x3bU, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))) 
           & ((0x3aU >= (0x3fU & (VL_EXTENDS_II(6,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                  - (IData)(1U)))) 
              && (1U & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_mant 
                                >> (0x3fU & (VL_EXTENDS_II(6,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)) 
                                             - (IData)(1U))))))));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__round_up 
        = ((IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__guard_bit) 
           & ((IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__lsb_bit) 
              | (IData)((0ULL != bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sticky_masked))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sum_m_ext 
        = (VL_GTS_III(32, 0x3bU, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))
            ? (0x1ffU & (IData)((0x7ffffffffffffffULL 
                                 & VL_SHIFTR_QQI(59,59,9, vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_mant, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift)))))
            : 0U);
    if (bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__round_up) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sum_m_ext 
            = (0x1ffU & ((IData)(1U) + (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sum_m_ext)));
    }
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__lsb_bit 
        = (VL_GTS_III(32, 0x3bU, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift))) 
           & ((0x3aU >= (0x3fU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift))) 
              && (1U & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_mant 
                                >> (0x3fU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))))));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__guard_bit 
        = ((VL_LTS_III(32, 0U, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift))) 
            & VL_GTES_III(32, 0x3bU, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))) 
           & ((0x3aU >= (0x3fU & (VL_EXTENDS_II(6,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                  - (IData)(1U)))) 
              && (1U & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_mant 
                                >> (0x3fU & (VL_EXTENDS_II(6,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)) 
                                             - (IData)(1U))))))));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__round_up 
        = ((IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__guard_bit) 
           & ((IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__lsb_bit) 
              | (IData)((0ULL != bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sticky_masked))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sum_m_ext 
        = (VL_GTS_III(32, 0x3bU, VL_EXTENDS_II(32,9, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))
            ? (0x1ffU & (IData)((0x7ffffffffffffffULL 
                                 & VL_SHIFTR_QQI(59,59,9, vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_mant, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift)))))
            : 0U);
    if (bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__round_up) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sum_m_ext 
            = (0x1ffU & ((IData)(1U) + (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sum_m_ext)));
    }
    vlSelf->o_exp2_tready = (1U & ((~ (IData)(vlSelf->o_exp2_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_exp2opt_tready = (1U & ((~ (IData)(vlSelf->o_exp2opt_tvalid)) 
                                      | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_lut_tready = (1U & ((~ (IData)(vlSelf->o_lut_tvalid)) 
                                  | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_hybrid_tready = (1U & ((~ (IData)(vlSelf->o_hybrid_tvalid)) 
                                     | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_cut_tready = (1U & ((~ (IData)(vlSelf->o_cut_tvalid)) 
                                  | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_poly4_tready = (1U & ((~ (IData)(vlSelf->o_poly4_tvalid)) 
                                    | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_poly4dsp_tready = (1U & ((~ (IData)(vlSelf->o_poly4dsp_tvalid)) 
                                       | (IData)(vlSelf->m_axis_tready)));
    if ((3U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
         [9U])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (8U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (0x200000U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (0x800U | (0x3ff01fU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp));
    } else if ((1U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [9U])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (0x200fffU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (0x10U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
    } else if ((2U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [9U])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (1U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
    } else {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s7_rounded_fp;
    }
    if ((3U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
         [9U])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (8U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (0x200000U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (0x800U | (0x3ff01fU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp));
    } else if ((1U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [9U])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (0x200fffU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (0x10U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
    } else if ((2U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [9U])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (1U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
    } else {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s7_rounded_fp;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->s_axis_tready = ((IData)(vlSelf->o_exp2_tready) 
                             & ((IData)(vlSelf->o_exp2opt_tready) 
                                & ((IData)(vlSelf->o_lut_tready) 
                                   & ((IData)(vlSelf->o_hybrid_tready) 
                                      & ((IData)(vlSelf->o_cut_tready) 
                                         & ((IData)(vlSelf->o_poly4_tready) 
                                            & (IData)(vlSelf->o_poly4dsp_tready)))))));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0U;
    if (VL_ONEHOT0_I(((8U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                             << 2U)) | ((4U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                                               >> 1U)) 
                                        | ((2U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                                                  >> 1U)) 
                                           | (1U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp)))))) {
        if ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp)) {
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
        } else if ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp)) {
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
        } else if ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp)) {
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                = ((0U == (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                                    >> 5U))) ? 0x40U
                    : (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                                >> 5U)));
        } else if ((2U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp)) {
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                = (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                            >> 5U));
        } else {
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp 
                = (0x1ffU & ((IData)(0x7fU) + (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                                               >> 0xcU)));
            if (VL_GTES_III(9, 0U, (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp))) {
                bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
                bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
            } else if (VL_LTES_III(9, 0xffU, (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp))) {
                bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
                bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
            } else {
                bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp 
                    = (0xffU & (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp));
                bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                    = (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                                >> 5U));
            }
        }
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__bf16_comb 
        = ((0x8000U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
                       >> 6U)) | (((IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp) 
                                   << 7U) | (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out)));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0U;
    if (VL_ONEHOT0_I(((8U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                             << 2U)) | ((4U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                                               >> 1U)) 
                                        | ((2U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                                                  >> 1U)) 
                                           | (1U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp)))))) {
        if ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp)) {
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
        } else if ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp)) {
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
        } else if ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp)) {
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                = ((0U == (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                                    >> 5U))) ? 0x40U
                    : (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                                >> 5U)));
        } else if ((2U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp)) {
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                = (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                            >> 5U));
        } else {
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp 
                = (0x1ffU & ((IData)(0x7fU) + (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                                               >> 0xcU)));
            if (VL_GTES_III(9, 0U, (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp))) {
                bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
                bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
            } else if (VL_LTES_III(9, 0xffU, (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp))) {
                bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
                bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
            } else {
                bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp 
                    = (0xffU & (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp));
                bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                    = (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                                >> 5U));
            }
        }
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__bf16_comb 
        = ((0x8000U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
                       >> 6U)) | (((IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp) 
                                   << 7U) | (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out)));
}

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___eval_stl(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vbf16_expe_equiv_top___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___eval_triggers__stl(Vbf16_expe_equiv_top___024root* vlSelf);

VL_ATTR_COLD bool Vbf16_expe_equiv_top___024root___eval_phase__stl(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vbf16_expe_equiv_top___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vbf16_expe_equiv_top___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___dump_triggers__ico(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___dump_triggers__ico\n"); );
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
VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___dump_triggers__act(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___dump_triggers__act\n"); );
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
VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___dump_triggers__nba(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___dump_triggers__nba\n"); );
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

VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___ctor_var_reset(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->rst_n = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tdata = VL_RAND_RESET_I(16);
    vlSelf->s_axis_tvalid = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tready = VL_RAND_RESET_I(1);
    vlSelf->m_axis_tready = VL_RAND_RESET_I(1);
    vlSelf->o_exp2_tdata = VL_RAND_RESET_I(16);
    vlSelf->o_exp2_tvalid = VL_RAND_RESET_I(1);
    vlSelf->o_exp2_tready = VL_RAND_RESET_I(1);
    vlSelf->o_exp2opt_tdata = VL_RAND_RESET_I(16);
    vlSelf->o_exp2opt_tvalid = VL_RAND_RESET_I(1);
    vlSelf->o_exp2opt_tready = VL_RAND_RESET_I(1);
    vlSelf->o_lut_tdata = VL_RAND_RESET_I(16);
    vlSelf->o_lut_tvalid = VL_RAND_RESET_I(1);
    vlSelf->o_lut_tready = VL_RAND_RESET_I(1);
    vlSelf->o_hybrid_tdata = VL_RAND_RESET_I(16);
    vlSelf->o_hybrid_tvalid = VL_RAND_RESET_I(1);
    vlSelf->o_hybrid_tready = VL_RAND_RESET_I(1);
    vlSelf->o_cut_tdata = VL_RAND_RESET_I(16);
    vlSelf->o_cut_tvalid = VL_RAND_RESET_I(1);
    vlSelf->o_cut_tready = VL_RAND_RESET_I(1);
    vlSelf->o_poly4_tdata = VL_RAND_RESET_I(16);
    vlSelf->o_poly4_tvalid = VL_RAND_RESET_I(1);
    vlSelf->o_poly4_tready = VL_RAND_RESET_I(1);
    vlSelf->o_poly4dsp_tdata = VL_RAND_RESET_I(16);
    vlSelf->o_poly4dsp_tvalid = VL_RAND_RESET_I(1);
    vlSelf->o_poly4dsp_tready = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s2_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_base2 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_src = VL_RAND_RESET_I(8);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_out = VL_RAND_RESET_I(31);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_frac_part = VL_RAND_RESET_Q(38);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_norm_mant = VL_RAND_RESET_Q(59);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_poly_exp = VL_RAND_RESET_I(9);
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s7_rounded_fp = VL_RAND_RESET_I(22);
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[__Vi0] = VL_RAND_RESET_I(2);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__bf16_out = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__gen_axi_pipelined__DOT__vld_sr = VL_RAND_RESET_I(12);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__mant_mult = VL_RAND_RESET_I(31);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__mant_src_d = VL_RAND_RESET_I(8);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__base2_d = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_unified_shift__DOT__unified_shifted = VL_RAND_RESET_Q(47);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__packed_coeff = VL_RAND_RESET_Q(42);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__frac_aligned = VL_RAND_RESET_Q(38);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__ax_core = VL_RAND_RESET_Q(59);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__coeff_b_stage = VL_RAND_RESET_I(21);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__calc_res = VL_RAND_RESET_Q(62);
    for (int __Vi0 = 0; __Vi0 < 128; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[__Vi0] = VL_RAND_RESET_Q(42);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_comb = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_s = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__res_s = VL_RAND_RESET_Q(62);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__poly_mant_comb = VL_RAND_RESET_Q(59);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__final_exponent = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__shift_9 = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_final_exp = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_mant = VL_RAND_RESET_Q(59);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sum_m_ext = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_m_ext = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_base_exp = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__bf16_comb = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s2_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_base2 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_src = VL_RAND_RESET_I(8);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_out = VL_RAND_RESET_I(31);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s4_frac_part = VL_RAND_RESET_Q(38);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s4_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s6_norm_mant = VL_RAND_RESET_Q(59);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s6_poly_exp = VL_RAND_RESET_I(9);
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[__Vi0] = VL_RAND_RESET_I(9);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s7_rounded_fp = VL_RAND_RESET_I(22);
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[__Vi0] = VL_RAND_RESET_I(2);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__bf16_out = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__gen_axi_pipelined__DOT__vld_sr = VL_RAND_RESET_I(12);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__mant_mult = VL_RAND_RESET_I(31);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__mant_src_d = VL_RAND_RESET_I(8);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__base2_d = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__unified_shifted = VL_RAND_RESET_Q(47);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__packed_coeff = VL_RAND_RESET_Q(42);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__frac_aligned = VL_RAND_RESET_Q(38);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__ax_core = VL_RAND_RESET_Q(51);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__coeff_b_stage = VL_RAND_RESET_I(21);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__calc_res = VL_RAND_RESET_Q(62);
    for (int __Vi0 = 0; __Vi0 < 128; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom[__Vi0] = VL_RAND_RESET_Q(42);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_comb = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_s = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__res_s = VL_RAND_RESET_Q(62);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__poly_mant_comb = VL_RAND_RESET_Q(59);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__final_exponent = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__shift_9 = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_final_exp = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_mant = VL_RAND_RESET_Q(59);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sum_m_ext = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_m_ext = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_base_exp = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_is_sub = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__bf16_comb = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_addr = VL_RAND_RESET_I(12);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s3_rom_data = VL_RAND_RESET_I(16);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay[__Vi0] = VL_RAND_RESET_I(2);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__core_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__gen_axi_pipelined__DOT__vld_sr = VL_RAND_RESET_I(12);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 2176; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_sparse_exp_index = VL_RAND_RESET_I(3);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_mantissa = VL_RAND_RESET_I(7);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_dense_addr = VL_RAND_RESET_I(10);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_sparse_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_dense_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__core_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr = VL_RAND_RESET_I(12);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_sparse__DOT__data_comb = VL_RAND_RESET_I(16);
    for (int __Vi0 = 0; __Vi0 < 1024; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_prod = VL_RAND_RESET_I(32);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_shift = VL_RAND_RESET_I(5);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_tail_addr = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_frac = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_tail_addr = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_cand_dev = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_tail_data = VL_RAND_RESET_I(7);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_frac = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__base_count = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_base_count = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_cut_dev = VL_RAND_RESET_I(13);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_cut_sentinel = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_frac = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_tail_data = VL_RAND_RESET_I(7);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__ladder_count = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_below_cut = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_base_count = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_tail_data = VL_RAND_RESET_I(7);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__core_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__gen_axi_pipelined__DOT__valid_sr = VL_RAND_RESET_I(12);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT____VdfgTmp_h0f01a197__0 = 0;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(4);
    }
    for (int __Vi0 = 0; __Vi0 < 11; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(7);
    }
    for (int __Vi0 = 0; __Vi0 < 128; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(13);
    }
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_eo = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_frac = VL_RAND_RESET_I(17);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_tail_addr = VL_RAND_RESET_I(4);
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo[__Vi0] = VL_RAND_RESET_I(2);
    }
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route[__Vi0] = VL_RAND_RESET_I(2);
    }
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int[__Vi0] = VL_RAND_RESET_I(9);
    }
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail[__Vi0] = VL_RAND_RESET_I(7);
    }
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac[__Vi0] = VL_RAND_RESET_I(17);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__core_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_axi_pipelined__DOT__valid_sr = VL_RAND_RESET_I(12);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_prod = VL_RAND_RESET_I(32);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_shift = VL_RAND_RESET_I(5);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_tail = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_eo = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____VdfgTmp_h785166bd__0 = 0;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 11; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(7);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_horner__BRA__0__KET____DOT__u_step__DOT__mult_r = VL_RAND_RESET_Q(43);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_horner__BRA__1__KET____DOT__u_step__DOT__mult_r = VL_RAND_RESET_Q(43);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_horner__BRA__2__KET____DOT__u_step__DOT__mult_r = VL_RAND_RESET_Q(43);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_horner__BRA__3__KET____DOT__u_step__DOT__mult_r = VL_RAND_RESET_Q(43);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_eo = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_frac = VL_RAND_RESET_I(17);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_tail_addr = VL_RAND_RESET_I(4);
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[__Vi0] = VL_RAND_RESET_I(2);
    }
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[__Vi0] = VL_RAND_RESET_I(2);
    }
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[__Vi0] = VL_RAND_RESET_I(9);
    }
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[__Vi0] = VL_RAND_RESET_I(7);
    }
    for (int __Vi0 = 0; __Vi0 < 9; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[__Vi0] = VL_RAND_RESET_I(17);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__core_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_axi_pipelined__DOT__valid_sr = VL_RAND_RESET_I(12);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_t7 = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_onehot = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_tail = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_eo = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__scaled_comb = VL_RAND_RESET_Q(41);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 11; ++__Vi0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(7);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__0__KET____DOT__u_step__DOT__mult_r = VL_RAND_RESET_Q(43);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__1__KET____DOT__u_step__DOT__mult_r = VL_RAND_RESET_Q(43);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__2__KET____DOT__u_step__DOT__mult_r = VL_RAND_RESET_Q(43);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__3__KET____DOT__u_step__DOT__mult_r = VL_RAND_RESET_Q(43);
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v0 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v0 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v0 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v0 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v0 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v1 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v1 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v1 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v1 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v1 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v2 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v2 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v2 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v2 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v2 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v3 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v3 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v3 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v3 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v3 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v4 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v4 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v4 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v4 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v4 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v5 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v5 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v5 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v5 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v5 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v6 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v6 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v6 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v6 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v6 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v7 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v7 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v7 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v7 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v7 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v0 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v0 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v0 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v0 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v0 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v1 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v1 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v1 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v1 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v1 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v2 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v2 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v2 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v2 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v2 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v3 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v3 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v3 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v3 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v3 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v4 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v4 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v4 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v4 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v4 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v4 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v5 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v5 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v5 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v5 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v5 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v5 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v6 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v6 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v6 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v6 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v6 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v6 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v7 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v7 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v7 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v7 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v7 = 0;
    vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v7 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v7 = 0;
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = VL_RAND_RESET_I(1);
}
