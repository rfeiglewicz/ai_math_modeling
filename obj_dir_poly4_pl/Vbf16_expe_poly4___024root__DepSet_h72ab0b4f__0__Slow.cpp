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
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = vlSelf->rst_n;
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

VL_ATTR_COLD void Vbf16_expe_poly4___024root___stl_sequent__TOP__0(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___stl_sequent__TOP__0\n"); );
    // Body
    vlSelf->m_axis_tdata = vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0U];
    vlSelf->bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0 
        = (((IData)(0xb8aa3bU) * (0x80U | (0x7fU & 
                                           (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                            >> 5U)))) 
           >> (0x1fU & ((IData)(0xdU) - (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                         >> 0xcU))));
    vlSelf->bf16_expe_poly4__DOT__ctrl_route[0U] = vlSelf->bf16_expe_poly4__DOT__s2_route;
    vlSelf->bf16_expe_poly4__DOT__ctrl_eo[0U] = vlSelf->bf16_expe_poly4__DOT__s2_eo;
    vlSelf->bf16_expe_poly4__DOT__ctrl_int[0U] = vlSelf->bf16_expe_poly4__DOT__s2_int_part;
    vlSelf->bf16_expe_poly4__DOT__ctrl_frac[0U] = vlSelf->bf16_expe_poly4__DOT__s2_frac;
    vlSelf->bf16_expe_poly4__DOT__ctrl_tail[0U] = (
                                                   (0xaU 
                                                    >= (IData)(vlSelf->bf16_expe_poly4__DOT__s2_tail_addr))
                                                    ? 
                                                   vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom
                                                   [vlSelf->bf16_expe_poly4__DOT__s2_tail_addr]
                                                    : 0U);
    vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb 
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
    vlSelf->m_axis_tvalid = (1U & ((IData)(vlSelf->bf16_expe_poly4__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                   >> 7U));
    vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
}

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_stl(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vbf16_expe_poly4___024root___stl_sequent__TOP__0(vlSelf);
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
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk or negedge rst_n)\n");
    }
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(posedge clk)\n");
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
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk or negedge rst_n)\n");
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(posedge clk)\n");
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
    vlSelf->bf16_expe_poly4__DOT__s1_decomposed = VL_RAND_RESET_I(22);
    vlSelf->bf16_expe_poly4__DOT__s2_route = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_poly4__DOT__s2_eo = VL_RAND_RESET_I(2);
    vlSelf->bf16_expe_poly4__DOT__s2_int_part = VL_RAND_RESET_I(9);
    vlSelf->bf16_expe_poly4__DOT__s2_frac = VL_RAND_RESET_I(17);
    vlSelf->bf16_expe_poly4__DOT__s2_tail_addr = VL_RAND_RESET_I(4);
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[__Vi0] = VL_RAND_RESET_I(2);
    }
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[__Vi0] = VL_RAND_RESET_I(2);
    }
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[__Vi0] = VL_RAND_RESET_I(9);
    }
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[__Vi0] = VL_RAND_RESET_I(7);
    }
    for (int __Vi0 = 0; __Vi0 < 5; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[__Vi0] = VL_RAND_RESET_I(17);
    }
    vlSelf->bf16_expe_poly4__DOT__core_data = VL_RAND_RESET_I(16);
    vlSelf->bf16_expe_poly4__DOT__gen_axi_pipelined__DOT__valid_sr = VL_RAND_RESET_I(8);
    vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out = VL_RAND_RESET_I(25);
    vlSelf->bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0 = 0;
    vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb = VL_RAND_RESET_I(4);
    vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 11; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom[__Vi0] = VL_RAND_RESET_I(7);
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v0 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v0 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v0 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v0 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v0 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v0 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v1 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v1 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v1 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v1 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v1 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v1 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v2 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v2 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v2 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v2 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v2 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v2 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v3 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v3 = VL_RAND_RESET_I(2);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v3 = VL_RAND_RESET_I(9);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v3 = VL_RAND_RESET_I(7);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v3 = 0;
    vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v3 = VL_RAND_RESET_I(17);
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v3 = 0;
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = VL_RAND_RESET_I(1);
}
