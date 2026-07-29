// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_poly4.h for the primary calling header

#include "Vbf16_expe_poly4__pch.h"
#include "Vbf16_expe_poly4___024root.h"

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_static(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_initial__TOP(Vbf16_expe_poly4___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_initial(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_initial\n"); );
    // Body
    Vbf16_expe_poly4___024root___eval_initial__TOP(vlSelf);
}

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_initial__TOP(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_initial__TOP\n"); );
    // Body
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[0U] = 0x6dU;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[1U] = 0x42U;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[2U] = 0x28U;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[3U] = 0x18U;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[4U] = 0xfU;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[5U] = 9U;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[6U] = 5U;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[7U] = 3U;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[8U] = 2U;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[9U] = 1U;
    vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[0xaU] = 1U;
}

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_final(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__stl(Vbf16_expe_poly4___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vbf16_expe_poly4___024root___eval_phase__stl(Vbf16_expe_poly4___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_settle(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_settle\n"); );
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
            Vbf16_expe_poly4___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_poly4.sv", 96, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vbf16_expe_poly4___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__stl(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

void Vbf16_expe_poly4___024root___ico_sequent__TOP__0(Vbf16_expe_poly4___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_stl(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vbf16_expe_poly4___024root___ico_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_triggers__stl(Vbf16_expe_poly4___024root* vlSelf);

VL_ATTR_COLD bool Vbf16_expe_poly4___024root___eval_phase__stl(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vbf16_expe_poly4___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vbf16_expe_poly4___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__ico(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___dump_triggers__ico\n"); );
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
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__act(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__nba(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vbf16_expe_poly4___024root___ctor_var_reset(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->rst_n = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tdata = VL_RAND_RESET_I(16);
    vlSelf->s_axis_tvalid = VL_RAND_RESET_I(1);
    vlSelf->s_axis_tready = VL_RAND_RESET_I(1);
    vlSelf->m_axis_tdata = VL_RAND_RESET_I(16);
    vlSelf->m_axis_tvalid = VL_RAND_RESET_I(1);
    vlSelf->m_axis_tready = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[__Vi0] = VL_RAND_RESET_I(2);
    }
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[__Vi0] = VL_RAND_RESET_I(2);
    }
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[__Vi0] = VL_RAND_RESET_I(7);
    }
    for (int __Vi0 = 0; __Vi0 < 11; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(7);
    }
}
