// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_poly4.h for the primary calling header

#include "Vbf16_expe_poly4__pch.h"
#include "Vbf16_expe_poly4___024root.h"

VL_INLINE_OPT void Vbf16_expe_poly4___024root___ico_sequent__TOP__0(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___ico_sequent__TOP__0\n"); );
    // Init
    IData/*21:0*/ bf16_expe_poly4__DOT__s1_decomposed;
    bf16_expe_poly4__DOT__s1_decomposed = 0;
    IData/*24:0*/ bf16_expe_poly4__DOT____Vcellinp__gen_horner__BRA__2__KET____DOT__u_step__acc_in;
    bf16_expe_poly4__DOT____Vcellinp__gen_horner__BRA__2__KET____DOT__u_step__acc_in = 0;
    IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out;
    bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out = 0;
    IData/*24:0*/ bf16_expe_poly4__DOT____Vcellinp__gen_horner__BRA__3__KET____DOT__u_step__acc_in;
    bf16_expe_poly4__DOT____Vcellinp__gen_horner__BRA__3__KET____DOT__u_step__acc_in = 0;
    IData/*31:0*/ bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0;
    bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0 = 0;
    CData/*3:0*/ bf16_expe_poly4__DOT__u_decompose__DOT__status_comb;
    bf16_expe_poly4__DOT__u_decompose__DOT__status_comb = 0;
    SData/*8:0*/ bf16_expe_poly4__DOT__u_decompose__DOT__exponent_comb;
    bf16_expe_poly4__DOT__u_decompose__DOT__exponent_comb = 0;
    CData/*0:0*/ bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb;
    bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb = 0;
    CData/*1:0*/ bf16_expe_poly4__DOT__u_early_out__DOT__eo_comb;
    bf16_expe_poly4__DOT__u_early_out__DOT__eo_comb = 0;
    // Body
    vlSelf->m_axis_tvalid = vlSelf->s_axis_tvalid;
    vlSelf->s_axis_tready = vlSelf->m_axis_tready;
    if ((0xaU >= (0xfU & ((IData)(vlSelf->s_axis_tdata) 
                          - (IData)(0xfU))))) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[0U] 
            = vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom
            [(0xfU & ((IData)(vlSelf->s_axis_tdata) 
                      - (IData)(0xfU)))];
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[1U] 
            = vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom
            [(0xfU & ((IData)(vlSelf->s_axis_tdata) 
                      - (IData)(0xfU)))];
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[2U] 
            = vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom
            [(0xfU & ((IData)(vlSelf->s_axis_tdata) 
                      - (IData)(0xfU)))];
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[3U] 
            = vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom
            [(0xfU & ((IData)(vlSelf->s_axis_tdata) 
                      - (IData)(0xfU)))];
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[4U] 
            = vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom
            [(0xfU & ((IData)(vlSelf->s_axis_tdata) 
                      - (IData)(0xfU)))];
    } else {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[0U] = 0U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[1U] = 0U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[2U] = 0U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[3U] = 0U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[4U] = 0U;
    }
    bf16_expe_poly4__DOT__u_decompose__DOT__status_comb 
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
    bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(bf16_expe_poly4__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(bf16_expe_poly4__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    bf16_expe_poly4__DOT__u_decompose__DOT__exponent_comb 
        = ((IData)((0U != (0xdU & (IData)(bf16_expe_poly4__DOT__u_decompose__DOT__status_comb))))
            ? 0U : ((2U & (IData)(bf16_expe_poly4__DOT__u_decompose__DOT__status_comb))
                     ? 0x182U : (0x1ffU & ((0xffU & 
                                            ((IData)(vlSelf->s_axis_tdata) 
                                             >> 7U)) 
                                           - (IData)(0x7fU)))));
    if (VL_LTS_III(9, 6U, (IData)(bf16_expe_poly4__DOT__u_decompose__DOT__exponent_comb))) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[0U] = 2U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[1U] = 2U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[2U] = 2U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[3U] = 2U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[4U] = 2U;
    } else if (((6U == (IData)(bf16_expe_poly4__DOT__u_decompose__DOT__exponent_comb)) 
                & (0x3aU <= (0x7fU & (IData)(vlSelf->s_axis_tdata))))) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[0U] = 2U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[1U] = 2U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[2U] = 2U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[3U] = 2U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[4U] = 2U;
    } else if (((6U == (IData)(bf16_expe_poly4__DOT__u_decompose__DOT__exponent_comb)) 
                & (0x2fU <= (0x7fU & (IData)(vlSelf->s_axis_tdata))))) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[0U] = 1U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[1U] = 1U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[2U] = 1U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[3U] = 1U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[4U] = 1U;
    } else {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[0U] = 0U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[1U] = 0U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[2U] = 0U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[3U] = 0U;
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[4U] = 0U;
    }
    bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0 = 
        (((IData)(0xb8aa3bU) * (0x80U | (0x7fU & (IData)(vlSelf->s_axis_tdata)))) 
         >> (0x1fU & ((IData)(0xdU) - (IData)(bf16_expe_poly4__DOT__u_decompose__DOT__exponent_comb))));
    bf16_expe_poly4__DOT__s1_decomposed = ((0x200000U 
                                            & ((IData)(vlSelf->s_axis_tdata) 
                                               << 6U)) 
                                           | (((IData)(bf16_expe_poly4__DOT__u_decompose__DOT__exponent_comb) 
                                               << 0xcU) 
                                              | ((0xfe0U 
                                                  & ((IData)(vlSelf->s_axis_tdata) 
                                                     << 5U)) 
                                                 | (((IData)(bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                     << 4U) 
                                                    | (IData)(bf16_expe_poly4__DOT__u_decompose__DOT__status_comb)))));
    bf16_expe_poly4__DOT____Vcellinp__gen_horner__BRA__2__KET____DOT__u_step__acc_in 
        = (0x1ffffffU & (IData)((0x7ffffffffffULL & 
                                 VL_SHIFTRS_QQI(43,43,32, 
                                                (0x7ffffffffffULL 
                                                 & (0x3d54010000ULL 
                                                    + 
                                                    VL_MULS_QQQ(43, 
                                                                (0x7ffffffffffULL 
                                                                 & VL_EXTENDS_QI(43,25, 
                                                                                (0x1ffffffU 
                                                                                & (IData)(
                                                                                (0x7ffffffffffULL 
                                                                                & VL_SHIFTRS_QQI(43,43,32, 
                                                                                (0x7ffffffffffULL 
                                                                                & (0x7f252010000ULL 
                                                                                + 
                                                                                VL_MULS_QQQ(43, 0xe400ULL, 
                                                                                (0x7ffffffffffULL 
                                                                                & VL_EXTENDS_QI(43,18, 
                                                                                (0x1ffffU 
                                                                                & bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0)))))), 0x11U)))))), 
                                                                (0x7ffffffffffULL 
                                                                 & VL_EXTENDS_QI(43,18, 
                                                                                (0x1ffffU 
                                                                                & bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0)))))), 0x11U))));
    bf16_expe_poly4__DOT__u_early_out__DOT__eo_comb 
        = ((8U & bf16_expe_poly4__DOT__s1_decomposed)
            ? 3U : ((1U & bf16_expe_poly4__DOT__s1_decomposed)
                     ? 1U : ((4U & bf16_expe_poly4__DOT__s1_decomposed)
                              ? ((0x200000U & bf16_expe_poly4__DOT__s1_decomposed)
                                  ? 2U : 1U) : ((0x200000U 
                                                 & bf16_expe_poly4__DOT__s1_decomposed)
                                                 ? 
                                                (VL_GTS_III(9, 0x1f7U, 
                                                            (0x1ffU 
                                                             & (bf16_expe_poly4__DOT__s1_decomposed 
                                                                >> 0xcU)))
                                                  ? 1U
                                                  : 
                                                 (VL_LTS_III(9, 7U, 
                                                             (0x1ffU 
                                                              & (bf16_expe_poly4__DOT__s1_decomposed 
                                                                 >> 0xcU)))
                                                   ? 2U
                                                   : 0U))
                                                 : 1U))));
    bf16_expe_poly4__DOT____Vcellinp__gen_horner__BRA__3__KET____DOT__u_step__acc_in 
        = (0x1ffffffU & (IData)((0x7ffffffffffULL & 
                                 VL_SHIFTRS_QQI(43,43,32, 
                                                (0x7ffffffffffULL 
                                                 & (0x74e92010000ULL 
                                                    + 
                                                    VL_MULS_QQQ(43, 
                                                                (0x7ffffffffffULL 
                                                                 & VL_EXTENDS_QI(43,25, bf16_expe_poly4__DOT____Vcellinp__gen_horner__BRA__2__KET____DOT__u_step__acc_in)), 
                                                                (0x7ffffffffffULL 
                                                                 & VL_EXTENDS_QI(43,18, 
                                                                                (0x1ffffU 
                                                                                & bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0)))))), 0x11U))));
    vlSelf->bf16_expe_poly4__DOT__ctrl_eo[0U] = bf16_expe_poly4__DOT__u_early_out__DOT__eo_comb;
    vlSelf->bf16_expe_poly4__DOT__ctrl_eo[1U] = bf16_expe_poly4__DOT__u_early_out__DOT__eo_comb;
    vlSelf->bf16_expe_poly4__DOT__ctrl_eo[2U] = bf16_expe_poly4__DOT__u_early_out__DOT__eo_comb;
    vlSelf->bf16_expe_poly4__DOT__ctrl_eo[3U] = bf16_expe_poly4__DOT__u_early_out__DOT__eo_comb;
    vlSelf->bf16_expe_poly4__DOT__ctrl_eo[4U] = bf16_expe_poly4__DOT__u_early_out__DOT__eo_comb;
    bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out 
        = (0x1ffffffU & (IData)((0x7ffffffffffULL & 
                                 VL_SHIFTRS_QQI(43,43,32, 
                                                (0x7ffffffffffULL 
                                                 & (0x68080010000ULL 
                                                    + 
                                                    VL_MULS_QQQ(43, 
                                                                (0x7ffffffffffULL 
                                                                 & VL_EXTENDS_QI(43,25, bf16_expe_poly4__DOT____Vcellinp__gen_horner__BRA__3__KET____DOT__u_step__acc_in)), 
                                                                (0x7ffffffffffULL 
                                                                 & VL_EXTENDS_QI(43,18, 
                                                                                (0x1ffffU 
                                                                                & bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0)))))), 0x11U))));
    vlSelf->m_axis_tdata = ((3U == vlSelf->bf16_expe_poly4__DOT__ctrl_eo
                             [4U]) ? 0xffc0U : ((1U 
                                                 == 
                                                 vlSelf->bf16_expe_poly4__DOT__ctrl_eo
                                                 [4U])
                                                 ? 0x3f80U
                                                 : 
                                                ((2U 
                                                  == 
                                                  vlSelf->bf16_expe_poly4__DOT__ctrl_eo
                                                  [4U])
                                                  ? 0U
                                                  : 
                                                 ((0U 
                                                   == 
                                                   vlSelf->bf16_expe_poly4__DOT__ctrl_route
                                                   [4U])
                                                   ? 
                                                  ((0x7f80U 
                                                    & ((((IData)(0x7eU) 
                                                         - 
                                                         (bf16_expe_poly4__DOT____VdfgTmp_hfc587e27__0 
                                                          >> 0x11U)) 
                                                        + 
                                                        VL_SHIFTR_III(8,8,32, 
                                                                      (0xffU 
                                                                       & VL_SHIFTRS_III(25,25,32, bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out, 0xfU)), 7U)) 
                                                       << 7U)) 
                                                   | (0x7fU 
                                                      & VL_SHIFTRS_III(25,25,32, bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out, 0xfU)))
                                                   : 
                                                  ((1U 
                                                    == 
                                                    vlSelf->bf16_expe_poly4__DOT__ctrl_route
                                                    [4U])
                                                    ? 
                                                   vlSelf->bf16_expe_poly4__DOT__ctrl_tail
                                                   [4U]
                                                    : 0U)))));
}

void Vbf16_expe_poly4___024root___eval_ico(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vbf16_expe_poly4___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vbf16_expe_poly4___024root___eval_triggers__ico(Vbf16_expe_poly4___024root* vlSelf);

bool Vbf16_expe_poly4___024root___eval_phase__ico(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vbf16_expe_poly4___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vbf16_expe_poly4___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vbf16_expe_poly4___024root___eval_act(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_act\n"); );
}

void Vbf16_expe_poly4___024root___eval_nba(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_nba\n"); );
}

void Vbf16_expe_poly4___024root___eval_triggers__act(Vbf16_expe_poly4___024root* vlSelf);

bool Vbf16_expe_poly4___024root___eval_phase__act(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<0> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vbf16_expe_poly4___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vbf16_expe_poly4___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vbf16_expe_poly4___024root___eval_phase__nba(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vbf16_expe_poly4___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__ico(Vbf16_expe_poly4___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__nba(Vbf16_expe_poly4___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__act(Vbf16_expe_poly4___024root* vlSelf);
#endif  // VL_DEBUG

void Vbf16_expe_poly4___024root___eval(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VicoIterCount;
    CData/*0:0*/ __VicoContinue;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VicoIterCount = 0U;
    vlSelf->__VicoFirstIteration = 1U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        if (VL_UNLIKELY((0x64U < __VicoIterCount))) {
#ifdef VL_DEBUG
            Vbf16_expe_poly4___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_poly4.sv", 96, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vbf16_expe_poly4___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vbf16_expe_poly4___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_poly4.sv", 96, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vbf16_expe_poly4___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("src/rtl/bf16_expe_poly4.sv", 96, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vbf16_expe_poly4___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vbf16_expe_poly4___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vbf16_expe_poly4___024root___eval_debug_assertions(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->rst_n & 0xfeU))) {
        Verilated::overWidthError("rst_n");}
    if (VL_UNLIKELY((vlSelf->s_axis_tvalid & 0xfeU))) {
        Verilated::overWidthError("s_axis_tvalid");}
    if (VL_UNLIKELY((vlSelf->m_axis_tready & 0xfeU))) {
        Verilated::overWidthError("m_axis_tready");}
}
#endif  // VL_DEBUG
