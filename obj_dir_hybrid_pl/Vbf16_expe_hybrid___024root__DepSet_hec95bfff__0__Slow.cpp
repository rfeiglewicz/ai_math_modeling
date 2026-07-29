// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_hybrid.h for the primary calling header

#include "Vbf16_expe_hybrid__pch.h"
#include "Vbf16_expe_hybrid___024root.h"

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___eval_static(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___eval_initial__TOP(Vbf16_expe_hybrid___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___eval_initial(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_initial\n"); );
    // Body
    Vbf16_expe_hybrid___024root___eval_initial__TOP(vlSelf);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = vlSelf->rst_n;
}

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___eval_initial__TOP(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_initial__TOP\n"); );
    // Init
    IData/*31:0*/ __Vilp;
    // Body
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0U] = 0x3f1bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[1U] = 0x3f1bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[2U] = 0x3f1aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[3U] = 0x3f19U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[4U] = 0x3f19U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[5U] = 0x3f18U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[6U] = 0x3f18U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[7U] = 0x3f17U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[8U] = 0x3f16U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[9U] = 0x3f16U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xaU] = 0x3f15U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xbU] = 0x3f15U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xcU] = 0x3f14U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xdU] = 0x3f14U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xeU] = 0x3f13U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xfU] = 0x3f12U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x10U] = 0x3f12U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x11U] = 0x3f11U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x12U] = 0x3f11U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x13U] = 0x3f10U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x14U] = 0x3f10U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x15U] = 0x3f0fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x16U] = 0x3f0eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x17U] = 0x3f0eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x18U] = 0x3f0dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x19U] = 0x3f0dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1aU] = 0x3f0cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1bU] = 0x3f0cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1cU] = 0x3f0bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1dU] = 0x3f0bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1eU] = 0x3f0aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1fU] = 0x3f0aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x20U] = 0x3f09U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x21U] = 0x3f08U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x22U] = 0x3f08U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x23U] = 0x3f07U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x24U] = 0x3f07U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x25U] = 0x3f06U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x26U] = 0x3f06U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x27U] = 0x3f05U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x28U] = 0x3f05U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x29U] = 0x3f04U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2aU] = 0x3f04U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2bU] = 0x3f03U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2cU] = 0x3f03U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2dU] = 0x3f02U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2eU] = 0x3f02U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2fU] = 0x3f01U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x30U] = 0x3f01U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x31U] = 0x3f00U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x32U] = 0x3effU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x33U] = 0x3efeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x34U] = 0x3efdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x35U] = 0x3efcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x36U] = 0x3efbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x37U] = 0x3efbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x38U] = 0x3efaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x39U] = 0x3ef9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3aU] = 0x3ef8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3bU] = 0x3ef7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3cU] = 0x3ef6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3dU] = 0x3ef5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3eU] = 0x3ef4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3fU] = 0x3ef3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x40U] = 0x3ef2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x41U] = 0x3ef1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x42U] = 0x3ef0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x43U] = 0x3eefU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x44U] = 0x3eeeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x45U] = 0x3eedU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x46U] = 0x3eecU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x47U] = 0x3eebU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x48U] = 0x3eeaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x49U] = 0x3ee9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x4aU] = 0x3ee9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x4bU] = 0x3ee8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x4cU] = 0x3ee7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x4dU] = 0x3ee6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x4eU] = 0x3ee5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x4fU] = 0x3ee4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x50U] = 0x3ee3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x51U] = 0x3ee2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x52U] = 0x3ee1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x53U] = 0x3ee1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x54U] = 0x3ee0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x55U] = 0x3edfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x56U] = 0x3edeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x57U] = 0x3eddU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x58U] = 0x3edcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x59U] = 0x3edbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x5aU] = 0x3edaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x5bU] = 0x3edaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x5cU] = 0x3ed9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x5dU] = 0x3ed8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x5eU] = 0x3ed7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x5fU] = 0x3ed6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x60U] = 0x3ed5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x61U] = 0x3ed5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x62U] = 0x3ed4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x63U] = 0x3ed3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x64U] = 0x3ed2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x65U] = 0x3ed1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x66U] = 0x3ed0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x67U] = 0x3ed0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x68U] = 0x3ecfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x69U] = 0x3eceU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x6aU] = 0x3ecdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x6bU] = 0x3eccU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x6cU] = 0x3eccU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x6dU] = 0x3ecbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x6eU] = 0x3ecaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x6fU] = 0x3ec9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x70U] = 0x3ec9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x71U] = 0x3ec8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x72U] = 0x3ec7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x73U] = 0x3ec6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x74U] = 0x3ec5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x75U] = 0x3ec5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x76U] = 0x3ec4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x77U] = 0x3ec3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x78U] = 0x3ec2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x79U] = 0x3ec2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x7aU] = 0x3ec1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x7bU] = 0x3ec0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x7cU] = 0x3ebfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x7dU] = 0x3ebfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x7eU] = 0x3ebeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x7fU] = 0x3ebdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x80U] = 0x3ebcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x81U] = 0x3ebbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x82U] = 0x3eb9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x83U] = 0x3eb8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x84U] = 0x3eb7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x85U] = 0x3eb5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x86U] = 0x3eb4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x87U] = 0x3eb2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x88U] = 0x3eb1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x89U] = 0x3eb0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x8aU] = 0x3eaeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x8bU] = 0x3eadU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x8cU] = 0x3eabU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x8dU] = 0x3eaaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x8eU] = 0x3ea9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x8fU] = 0x3ea8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x90U] = 0x3ea6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x91U] = 0x3ea5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x92U] = 0x3ea4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x93U] = 0x3ea2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x94U] = 0x3ea1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x95U] = 0x3ea0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x96U] = 0x3e9fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x97U] = 0x3e9dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x98U] = 0x3e9cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x99U] = 0x3e9bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x9aU] = 0x3e9aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x9bU] = 0x3e99U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x9cU] = 0x3e97U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x9dU] = 0x3e96U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x9eU] = 0x3e95U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x9fU] = 0x3e94U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa0U] = 0x3e93U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa1U] = 0x3e92U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa2U] = 0x3e90U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa3U] = 0x3e8fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa4U] = 0x3e8eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa5U] = 0x3e8dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa6U] = 0x3e8cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa7U] = 0x3e8bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa8U] = 0x3e8aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xa9U] = 0x3e89U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xaaU] = 0x3e88U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xabU] = 0x3e87U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xacU] = 0x3e86U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xadU] = 0x3e85U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xaeU] = 0x3e83U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xafU] = 0x3e82U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb0U] = 0x3e81U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb1U] = 0x3e80U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb2U] = 0x3e7fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb3U] = 0x3e7dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb4U] = 0x3e7bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb5U] = 0x3e79U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb6U] = 0x3e77U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb7U] = 0x3e75U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb8U] = 0x3e73U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xb9U] = 0x3e71U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xbaU] = 0x3e6fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xbbU] = 0x3e6eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xbcU] = 0x3e6cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xbdU] = 0x3e6aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xbeU] = 0x3e68U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xbfU] = 0x3e66U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc0U] = 0x3e64U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc1U] = 0x3e63U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc2U] = 0x3e61U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc3U] = 0x3e5fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc4U] = 0x3e5dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc5U] = 0x3e5cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc6U] = 0x3e5aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc7U] = 0x3e58U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc8U] = 0x3e57U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xc9U] = 0x3e55U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xcaU] = 0x3e53U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xcbU] = 0x3e52U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xccU] = 0x3e50U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xcdU] = 0x3e4eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xceU] = 0x3e4dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xcfU] = 0x3e4bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd0U] = 0x3e4aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd1U] = 0x3e48U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd2U] = 0x3e47U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd3U] = 0x3e45U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd4U] = 0x3e43U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd5U] = 0x3e42U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd6U] = 0x3e40U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd7U] = 0x3e3fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd8U] = 0x3e3dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xd9U] = 0x3e3cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xdaU] = 0x3e3aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xdbU] = 0x3e39U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xdcU] = 0x3e38U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xddU] = 0x3e36U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xdeU] = 0x3e35U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xdfU] = 0x3e33U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe0U] = 0x3e32U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe1U] = 0x3e31U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe2U] = 0x3e2fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe3U] = 0x3e2eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe4U] = 0x3e2cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe5U] = 0x3e2bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe6U] = 0x3e2aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe7U] = 0x3e28U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe8U] = 0x3e27U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xe9U] = 0x3e26U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xeaU] = 0x3e25U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xebU] = 0x3e23U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xecU] = 0x3e22U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xedU] = 0x3e21U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xeeU] = 0x3e20U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xefU] = 0x3e1eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf0U] = 0x3e1dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf1U] = 0x3e1cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf2U] = 0x3e1bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf3U] = 0x3e19U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf4U] = 0x3e18U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf5U] = 0x3e17U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf6U] = 0x3e16U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf7U] = 0x3e15U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf8U] = 0x3e14U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xf9U] = 0x3e12U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xfaU] = 0x3e11U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xfbU] = 0x3e10U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xfcU] = 0x3e0fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xfdU] = 0x3e0eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xfeU] = 0x3e0dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0xffU] = 0x3e0cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x100U] = 0x3e0bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x101U] = 0x3e08U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x102U] = 0x3e06U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x103U] = 0x3e04U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x104U] = 0x3e02U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x105U] = 0x3e00U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x106U] = 0x3dfcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x107U] = 0x3df8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x108U] = 0x3df5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x109U] = 0x3df1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x10aU] = 0x3dedU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x10bU] = 0x3de9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x10cU] = 0x3de6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x10dU] = 0x3de2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x10eU] = 0x3ddfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x10fU] = 0x3ddbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x110U] = 0x3dd8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x111U] = 0x3dd5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x112U] = 0x3dd1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x113U] = 0x3dceU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x114U] = 0x3dcbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x115U] = 0x3dc8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x116U] = 0x3dc5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x117U] = 0x3dc1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x118U] = 0x3dbeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x119U] = 0x3dbcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x11aU] = 0x3db9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x11bU] = 0x3db6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x11cU] = 0x3db3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x11dU] = 0x3db0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x11eU] = 0x3dadU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x11fU] = 0x3dabU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x120U] = 0x3da8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x121U] = 0x3da6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x122U] = 0x3da3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x123U] = 0x3da0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x124U] = 0x3d9eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x125U] = 0x3d9bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x126U] = 0x3d99U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x127U] = 0x3d97U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x128U] = 0x3d94U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x129U] = 0x3d92U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x12aU] = 0x3d90U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x12bU] = 0x3d8eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x12cU] = 0x3d8bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x12dU] = 0x3d89U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x12eU] = 0x3d87U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x12fU] = 0x3d85U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x130U] = 0x3d83U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x131U] = 0x3d81U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x132U] = 0x3d7eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x133U] = 0x3d7aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x134U] = 0x3d76U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x135U] = 0x3d72U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x136U] = 0x3d6eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x137U] = 0x3d6bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x138U] = 0x3d67U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x139U] = 0x3d63U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x13aU] = 0x3d60U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x13bU] = 0x3d5cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x13cU] = 0x3d59U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x13dU] = 0x3d56U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x13eU] = 0x3d52U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x13fU] = 0x3d4fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x140U] = 0x3d4cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x141U] = 0x3d49U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x142U] = 0x3d46U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x143U] = 0x3d43U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x144U] = 0x3d40U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x145U] = 0x3d3dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x146U] = 0x3d3aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x147U] = 0x3d37U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x148U] = 0x3d34U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x149U] = 0x3d31U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x14aU] = 0x3d2eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x14bU] = 0x3d2cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x14cU] = 0x3d29U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x14dU] = 0x3d26U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x14eU] = 0x3d24U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x14fU] = 0x3d21U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x150U] = 0x3d1fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x151U] = 0x3d1cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x152U] = 0x3d1aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x153U] = 0x3d18U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x154U] = 0x3d15U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x155U] = 0x3d13U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x156U] = 0x3d11U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x157U] = 0x3d0eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x158U] = 0x3d0cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x159U] = 0x3d0aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x15aU] = 0x3d08U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x15bU] = 0x3d06U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x15cU] = 0x3d04U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x15dU] = 0x3d02U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x15eU] = 0x3cffU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x15fU] = 0x3cfbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x160U] = 0x3cf7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x161U] = 0x3cf4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x162U] = 0x3cf0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x163U] = 0x3cecU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x164U] = 0x3ce8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x165U] = 0x3ce5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x166U] = 0x3ce1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x167U] = 0x3cdeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x168U] = 0x3cdaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x169U] = 0x3cd7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x16aU] = 0x3cd4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x16bU] = 0x3cd0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x16cU] = 0x3ccdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x16dU] = 0x3ccaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x16eU] = 0x3cc7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x16fU] = 0x3cc4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x170U] = 0x3cc1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x171U] = 0x3cbeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x172U] = 0x3cbbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x173U] = 0x3cb8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x174U] = 0x3cb5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x175U] = 0x3cb2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x176U] = 0x3cafU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x177U] = 0x3cadU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x178U] = 0x3caaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x179U] = 0x3ca7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x17aU] = 0x3ca5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x17bU] = 0x3ca2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x17cU] = 0x3ca0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x17dU] = 0x3c9dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x17eU] = 0x3c9bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x17fU] = 0x3c98U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x180U] = 0x3c96U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x181U] = 0x3c91U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x182U] = 0x3c8dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x183U] = 0x3c89U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x184U] = 0x3c84U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x185U] = 0x3c80U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x186U] = 0x3c79U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x187U] = 0x3c71U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x188U] = 0x3c6aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x189U] = 0x3c63U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x18aU] = 0x3c5cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x18bU] = 0x3c55U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x18cU] = 0x3c4eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x18dU] = 0x3c48U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x18eU] = 0x3c42U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x18fU] = 0x3c3cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x190U] = 0x3c36U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x191U] = 0x3c30U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x192U] = 0x3c2bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x193U] = 0x3c26U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x194U] = 0x3c21U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x195U] = 0x3c1cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x196U] = 0x3c17U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x197U] = 0x3c12U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x198U] = 0x3c0eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x199U] = 0x3c09U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x19aU] = 0x3c05U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x19bU] = 0x3c01U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x19cU] = 0x3bfaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x19dU] = 0x3bf2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x19eU] = 0x3bebU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x19fU] = 0x3be4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a0U] = 0x3bddU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a1U] = 0x3bd6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a2U] = 0x3bcfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a3U] = 0x3bc9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a4U] = 0x3bc3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a5U] = 0x3bbdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a6U] = 0x3bb7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a7U] = 0x3bb1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a8U] = 0x3bacU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1a9U] = 0x3ba7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1aaU] = 0x3ba2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1abU] = 0x3b9dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1acU] = 0x3b98U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1adU] = 0x3b93U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1aeU] = 0x3b8fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1afU] = 0x3b8aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b0U] = 0x3b86U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b1U] = 0x3b82U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b2U] = 0x3b7cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b3U] = 0x3b74U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b4U] = 0x3b6cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b5U] = 0x3b65U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b6U] = 0x3b5eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b7U] = 0x3b57U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b8U] = 0x3b51U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1b9U] = 0x3b4aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1baU] = 0x3b44U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1bbU] = 0x3b3eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1bcU] = 0x3b38U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1bdU] = 0x3b32U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1beU] = 0x3b2dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1bfU] = 0x3b28U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c0U] = 0x3b22U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c1U] = 0x3b1dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c2U] = 0x3b19U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c3U] = 0x3b14U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c4U] = 0x3b0fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c5U] = 0x3b0bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c6U] = 0x3b07U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c7U] = 0x3b03U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c8U] = 0x3afdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1c9U] = 0x3af5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1caU] = 0x3aeeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1cbU] = 0x3ae6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1ccU] = 0x3adfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1cdU] = 0x3ad8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1ceU] = 0x3ad2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1cfU] = 0x3acbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d0U] = 0x3ac5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d1U] = 0x3abfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d2U] = 0x3ab9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d3U] = 0x3ab3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d4U] = 0x3aaeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d5U] = 0x3aa9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d6U] = 0x3aa3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d7U] = 0x3a9eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d8U] = 0x3a99U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1d9U] = 0x3a95U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1daU] = 0x3a90U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1dbU] = 0x3a8cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1dcU] = 0x3a87U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1ddU] = 0x3a83U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1deU] = 0x3a7eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1dfU] = 0x3a77U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e0U] = 0x3a6fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e1U] = 0x3a68U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e2U] = 0x3a61U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e3U] = 0x3a5aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e4U] = 0x3a53U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e5U] = 0x3a4cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e6U] = 0x3a46U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e7U] = 0x3a40U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e8U] = 0x3a3aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1e9U] = 0x3a34U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1eaU] = 0x3a2fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1ebU] = 0x3a2aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1ecU] = 0x3a24U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1edU] = 0x3a1fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1eeU] = 0x3a1aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1efU] = 0x3a16U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f0U] = 0x3a11U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f1U] = 0x3a0dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f2U] = 0x3a08U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f3U] = 0x3a04U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f4U] = 0x3a00U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f5U] = 0x39f8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f6U] = 0x39f0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f7U] = 0x39e9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f8U] = 0x39e2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1f9U] = 0x39dbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1faU] = 0x39d4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1fbU] = 0x39ceU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1fcU] = 0x39c7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1fdU] = 0x39c1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1feU] = 0x39bbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x1ffU] = 0x39b5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x200U] = 0x39b0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x201U] = 0x39a5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x202U] = 0x399bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x203U] = 0x3992U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x204U] = 0x3989U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x205U] = 0x3981U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x206U] = 0x3972U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x207U] = 0x3963U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x208U] = 0x3955U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x209U] = 0x3948U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x20aU] = 0x393cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x20bU] = 0x3931U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x20cU] = 0x3926U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x20dU] = 0x391cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x20eU] = 0x3913U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x20fU] = 0x390aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x210U] = 0x3901U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x211U] = 0x38f3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x212U] = 0x38e4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x213U] = 0x38d7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x214U] = 0x38caU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x215U] = 0x38bdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x216U] = 0x38b2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x217U] = 0x38a7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x218U] = 0x389dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x219U] = 0x3893U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x21aU] = 0x388bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x21bU] = 0x3882U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x21cU] = 0x3875U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x21dU] = 0x3866U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x21eU] = 0x3858U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x21fU] = 0x384bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x220U] = 0x383eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x221U] = 0x3833U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x222U] = 0x3828U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x223U] = 0x381eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x224U] = 0x3814U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x225U] = 0x380bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x226U] = 0x3803U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x227U] = 0x37f6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x228U] = 0x37e7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x229U] = 0x37d9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x22aU] = 0x37ccU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x22bU] = 0x37bfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x22cU] = 0x37b4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x22dU] = 0x37a9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x22eU] = 0x379fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x22fU] = 0x3795U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x230U] = 0x378cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x231U] = 0x3784U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x232U] = 0x3777U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x233U] = 0x3768U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x234U] = 0x375aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x235U] = 0x374dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x236U] = 0x3741U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x237U] = 0x3735U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x238U] = 0x372aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x239U] = 0x3720U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x23aU] = 0x3716U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x23bU] = 0x370dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x23cU] = 0x3704U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x23dU] = 0x36f9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x23eU] = 0x36eaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x23fU] = 0x36dbU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x240U] = 0x36ceU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x241U] = 0x36c2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x242U] = 0x36b6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x243U] = 0x36abU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x244U] = 0x36a1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x245U] = 0x3697U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x246U] = 0x368eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x247U] = 0x3685U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x248U] = 0x367aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x249U] = 0x366bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x24aU] = 0x365dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x24bU] = 0x364fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x24cU] = 0x3643U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x24dU] = 0x3637U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x24eU] = 0x362cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x24fU] = 0x3621U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x250U] = 0x3618U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x251U] = 0x360eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x252U] = 0x3606U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x253U] = 0x35fcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x254U] = 0x35ecU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x255U] = 0x35deU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x256U] = 0x35d1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x257U] = 0x35c4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x258U] = 0x35b8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x259U] = 0x35adU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x25aU] = 0x35a2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x25bU] = 0x3599U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x25cU] = 0x358fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x25dU] = 0x3587U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x25eU] = 0x357dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x25fU] = 0x356eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x260U] = 0x355fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x261U] = 0x3552U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x262U] = 0x3545U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x263U] = 0x3539U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x264U] = 0x352eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x265U] = 0x3523U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x266U] = 0x3519U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x267U] = 0x3510U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x268U] = 0x3507U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x269U] = 0x34feU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x26aU] = 0x34efU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x26bU] = 0x34e0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x26cU] = 0x34d3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x26dU] = 0x34c6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x26eU] = 0x34baU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x26fU] = 0x34afU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x270U] = 0x34a4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x271U] = 0x349aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x272U] = 0x3491U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x273U] = 0x3488U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x274U] = 0x3480U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x275U] = 0x3470U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x276U] = 0x3462U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x277U] = 0x3454U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x278U] = 0x3447U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x279U] = 0x343bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x27aU] = 0x3430U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x27bU] = 0x3425U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x27cU] = 0x341bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x27dU] = 0x3412U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x27eU] = 0x3409U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x27fU] = 0x3401U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x280U] = 0x33f2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x281U] = 0x33d5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x282U] = 0x33bcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x283U] = 0x33a6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x284U] = 0x3393U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x285U] = 0x3381U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x286U] = 0x3364U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x287U] = 0x3349U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x288U] = 0x3332U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x289U] = 0x331dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x28aU] = 0x330aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x28bU] = 0x32f4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x28cU] = 0x32d8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x28dU] = 0x32beU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x28eU] = 0x32a8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x28fU] = 0x3294U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x290U] = 0x3283U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x291U] = 0x3267U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x292U] = 0x324cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x293U] = 0x3234U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x294U] = 0x321fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x295U] = 0x320cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x296U] = 0x31f7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x297U] = 0x31daU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x298U] = 0x31c1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x299U] = 0x31aaU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x29aU] = 0x3196U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x29bU] = 0x3184U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x29cU] = 0x316aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x29dU] = 0x314eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x29eU] = 0x3136U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x29fU] = 0x3121U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a0U] = 0x310eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a1U] = 0x30faU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a2U] = 0x30ddU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a3U] = 0x30c3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a4U] = 0x30acU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a5U] = 0x3098U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a6U] = 0x3086U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a7U] = 0x306cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a8U] = 0x3050U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2a9U] = 0x3038U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2aaU] = 0x3022U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2abU] = 0x300fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2acU] = 0x2ffdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2adU] = 0x2fdfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2aeU] = 0x2fc5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2afU] = 0x2faeU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b0U] = 0x2f99U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b1U] = 0x2f87U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b2U] = 0x2f6fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b3U] = 0x2f53U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b4U] = 0x2f3aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b5U] = 0x2f24U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b6U] = 0x2f11U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b7U] = 0x2f00U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b8U] = 0x2ee2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2b9U] = 0x2ec7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2baU] = 0x2eb0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2bbU] = 0x2e9bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2bcU] = 0x2e89U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2bdU] = 0x2e72U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2beU] = 0x2e55U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2bfU] = 0x2e3cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c0U] = 0x2e26U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c1U] = 0x2e13U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c2U] = 0x2e01U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c3U] = 0x2de4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c4U] = 0x2dc9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c5U] = 0x2db2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c6U] = 0x2d9dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c7U] = 0x2d8aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c8U] = 0x2d74U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2c9U] = 0x2d58U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2caU] = 0x2d3eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2cbU] = 0x2d28U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2ccU] = 0x2d14U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2cdU] = 0x2d03U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2ceU] = 0x2ce7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2cfU] = 0x2cccU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d0U] = 0x2cb4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d1U] = 0x2c9fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d2U] = 0x2c8cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d3U] = 0x2c77U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d4U] = 0x2c5aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d5U] = 0x2c40U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d6U] = 0x2c2aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d7U] = 0x2c16U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d8U] = 0x2c04U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2d9U] = 0x2be9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2daU] = 0x2bceU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2dbU] = 0x2bb6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2dcU] = 0x2ba0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2ddU] = 0x2b8eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2deU] = 0x2b7aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2dfU] = 0x2b5dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e0U] = 0x2b43U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e1U] = 0x2b2cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e2U] = 0x2b18U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e3U] = 0x2b06U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e4U] = 0x2aecU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e5U] = 0x2ad0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e6U] = 0x2ab8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e7U] = 0x2aa2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e8U] = 0x2a8fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2e9U] = 0x2a7dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2eaU] = 0x2a5fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2ebU] = 0x2a45U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2ecU] = 0x2a2eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2edU] = 0x2a19U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2eeU] = 0x2a07U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2efU] = 0x29efU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f0U] = 0x29d3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f1U] = 0x29baU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f2U] = 0x29a4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f3U] = 0x2991U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f4U] = 0x2980U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f5U] = 0x2962U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f6U] = 0x2947U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f7U] = 0x2930U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f8U] = 0x291bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2f9U] = 0x2909U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2faU] = 0x28f1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2fbU] = 0x28d5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2fcU] = 0x28bcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2fdU] = 0x28a6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2feU] = 0x2892U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x2ffU] = 0x2881U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x300U] = 0x2864U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x301U] = 0x2832U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x302U] = 0x280aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x303U] = 0x27d8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x304U] = 0x27a8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x305U] = 0x2783U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x306U] = 0x274cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x307U] = 0x271fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x308U] = 0x26f7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x309U] = 0x26c0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x30aU] = 0x2696U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x30bU] = 0x2669U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x30cU] = 0x2636U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x30dU] = 0x260eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x30eU] = 0x25dcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x30fU] = 0x25acU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x310U] = 0x2586U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x311U] = 0x2550U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x312U] = 0x2522U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x313U] = 0x24fdU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x314U] = 0x24c5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x315U] = 0x2499U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x316U] = 0x246fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x317U] = 0x243aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x318U] = 0x2411U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x319U] = 0x23e1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x31aU] = 0x23b0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x31bU] = 0x2389U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x31cU] = 0x2355U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x31dU] = 0x2326U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x31eU] = 0x2301U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x31fU] = 0x22c9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x320U] = 0x229dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x321U] = 0x2274U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x322U] = 0x223eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x323U] = 0x2214U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x324U] = 0x21e7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x325U] = 0x21b4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x326U] = 0x218cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x327U] = 0x215aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x328U] = 0x212aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x329U] = 0x2104U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x32aU] = 0x20ceU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x32bU] = 0x20a0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x32cU] = 0x207aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x32dU] = 0x2042U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x32eU] = 0x2017U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x32fU] = 0x1fecU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x330U] = 0x1fb8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x331U] = 0x1f8fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x332U] = 0x1f5fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x333U] = 0x1f2eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x334U] = 0x1f07U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x335U] = 0x1ed3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x336U] = 0x1ea4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x337U] = 0x1e7fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x338U] = 0x1e47U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x339U] = 0x1e1bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x33aU] = 0x1df1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x33bU] = 0x1dbcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x33cU] = 0x1d92U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x33dU] = 0x1d64U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x33eU] = 0x1d32U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x33fU] = 0x1d0aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x340U] = 0x1cd7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x341U] = 0x1ca8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x342U] = 0x1c83U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x343U] = 0x1c4bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x344U] = 0x1c1eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x345U] = 0x1bf7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x346U] = 0x1bc0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x347U] = 0x1b96U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x348U] = 0x1b69U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x349U] = 0x1b36U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x34aU] = 0x1b0dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x34bU] = 0x1adcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x34cU] = 0x1aacU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x34dU] = 0x1a86U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x34eU] = 0x1a50U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x34fU] = 0x1a22U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x350U] = 0x19fcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x351U] = 0x19c5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x352U] = 0x1999U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x353U] = 0x196eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x354U] = 0x193aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x355U] = 0x1911U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x356U] = 0x18e1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x357U] = 0x18afU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x358U] = 0x1889U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x359U] = 0x1855U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x35aU] = 0x1826U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x35bU] = 0x1801U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x35cU] = 0x17c9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x35dU] = 0x179dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x35eU] = 0x1774U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x35fU] = 0x173eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x360U] = 0x1714U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x361U] = 0x16e6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x362U] = 0x16b3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x363U] = 0x168cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x364U] = 0x165aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x365U] = 0x162aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x366U] = 0x1604U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x367U] = 0x15ceU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x368U] = 0x15a0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x369U] = 0x157aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x36aU] = 0x1542U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x36bU] = 0x1517U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x36cU] = 0x14ecU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x36dU] = 0x14b8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x36eU] = 0x148fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x36fU] = 0x145fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x370U] = 0x142dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x371U] = 0x1407U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x372U] = 0x13d2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x373U] = 0x13a4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x374U] = 0x137fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x375U] = 0x1347U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x376U] = 0x131bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x377U] = 0x12f1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x378U] = 0x12bcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x379U] = 0x1292U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x37aU] = 0x1264U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x37bU] = 0x1231U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x37cU] = 0x120aU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x37dU] = 0x11d7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x37eU] = 0x11a8U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x37fU] = 0x1183U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x380U] = 0x114bU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x381U] = 0x10f7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x382U] = 0x1096U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x383U] = 0x1035U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x384U] = 0xfdcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x385U] = 0xf86U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x386U] = 0xf22U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x387U] = 0xec4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x388U] = 0xe6eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x389U] = 0xe11U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x38aU] = 0xdafU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x38bU] = 0xd55U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x38cU] = 0xd01U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x38dU] = 0xc9cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x38eU] = 0xc3eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x38fU] = 0xbe6U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x390U] = 0xb8cU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x391U] = 0xb29U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x392U] = 0xaceU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x393U] = 0xa79U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x394U] = 0xa17U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x395U] = 0x9b7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x396U] = 0x95fU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x397U] = 0x907U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x398U] = 0x8a4U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x399U] = 0x847U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x39aU] = 0x7f1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x39bU] = 0x792U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x39cU] = 0x731U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x39dU] = 0x6d7U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x39eU] = 0x682U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x39fU] = 0x61eU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a0U] = 0x5c0U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a1U] = 0x569U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a2U] = 0x50dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a3U] = 0x4abU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a4U] = 0x450U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a5U] = 0x3fcU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a6U] = 0x399U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a7U] = 0x339U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a8U] = 0x2e1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3a9U] = 0x288U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3aaU] = 0x226U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3abU] = 0x1c9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3acU] = 0x174U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3adU] = 0x114U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3aeU] = 0xb3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3afU] = 0x6dU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b0U] = 0x42U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b1U] = 0x28U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b2U] = 0x18U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b3U] = 0xfU;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b4U] = 9U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b5U] = 5U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b6U] = 3U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b7U] = 2U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b8U] = 1U;
    vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[0x3b9U] = 1U;
    __Vilp = 0x3baU;
    while ((__Vilp <= 0x3ffU)) {
        vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[__Vilp] = 0U;
        __Vilp = ((IData)(1U) + __Vilp);
    }
}

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___eval_final(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_hybrid___024root___dump_triggers__stl(Vbf16_expe_hybrid___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vbf16_expe_hybrid___024root___eval_phase__stl(Vbf16_expe_hybrid___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___eval_settle(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_settle\n"); );
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
            Vbf16_expe_hybrid___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_hybrid.sv", 12, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vbf16_expe_hybrid___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_hybrid___024root___dump_triggers__stl(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

extern const VlUnpacked<SData/*15:0*/, 1024> Vbf16_expe_hybrid__ConstPool__TABLE_hf7ff168e_0;

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___stl_sequent__TOP__0(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___stl_sequent__TOP__0\n"); );
    // Init
    SData/*9:0*/ __Vtableidx1;
    __Vtableidx1 = 0;
    // Body
    vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb 
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
    __Vtableidx1 = (((IData)(vlSelf->bf16_expe_hybrid__DOT__s2_mantissa) 
                     << 3U) | (IData)(vlSelf->bf16_expe_hybrid__DOT__s2_sparse_exp_index));
    vlSelf->bf16_expe_hybrid__DOT__u_sparse__DOT__data_comb 
        = Vbf16_expe_hybrid__ConstPool__TABLE_hf7ff168e_0
        [__Vtableidx1];
    vlSelf->m_axis_tvalid = (1U & ((IData)(vlSelf->bf16_expe_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                   >> 3U));
    vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
}

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___eval_stl(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vbf16_expe_hybrid___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___eval_triggers__stl(Vbf16_expe_hybrid___024root* vlSelf);

VL_ATTR_COLD bool Vbf16_expe_hybrid___024root___eval_phase__stl(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vbf16_expe_hybrid___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vbf16_expe_hybrid___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_hybrid___024root___dump_triggers__ico(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___dump_triggers__ico\n"); );
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
VL_ATTR_COLD void Vbf16_expe_hybrid___024root___dump_triggers__act(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___dump_triggers__act\n"); );
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
VL_ATTR_COLD void Vbf16_expe_hybrid___024root___dump_triggers__nba(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___dump_triggers__nba\n"); );
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

VL_ATTR_COLD void Vbf16_expe_hybrid___024root___ctor_var_reset(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->rst_n = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tdata = VL_RAND_RESET_I(16);
    vlSelf->s_axis_tvalid = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tready = VL_RAND_RESET_I(1);
    vlSelf->m_axis_tdata = VL_RAND_RESET_I(16);
    vlSelf->m_axis_tvalid = VL_RAND_RESET_I(1);
    vlSelf->m_axis_tready = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_hybrid__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_hybrid__DOT__s2_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_hybrid__DOT__s2_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_hybrid__DOT__s2_sparse_exp_index = VL_RAND_RESET_I(3);
    vlSelf->bf16_expe_hybrid__DOT__s2_mantissa = VL_RAND_RESET_I(7);
    vlSelf->bf16_expe_hybrid__DOT__s2_dense_addr = VL_RAND_RESET_I(10);
    vlSelf->bf16_expe_hybrid__DOT__s3_sparse_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_hybrid__DOT__s3_dense_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_hybrid__DOT__s3_eo_code = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_hybrid__DOT__s3_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    vlSelf->bf16_expe_hybrid__DOT__u_sparse__DOT__data_comb = VL_RAND_RESET_I(16);
    for (int __Vi0 = 0; __Vi0 < 1024; ++__Vi0) {
        vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = VL_RAND_RESET_I(1);
}
