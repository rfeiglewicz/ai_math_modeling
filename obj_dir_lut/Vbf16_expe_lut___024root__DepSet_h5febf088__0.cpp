// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_lut.h for the primary calling header

#include "Vbf16_expe_lut__pch.h"
#include "Vbf16_expe_lut___024root.h"

VL_INLINE_OPT void Vbf16_expe_lut___024root___ico_sequent__TOP__0(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___ico_sequent__TOP__0\n"); );
    // Init
    IData/*21:0*/ bf16_expe_lut__DOT__s1_decomposed;
    bf16_expe_lut__DOT__s1_decomposed = 0;
    SData/*11:0*/ bf16_expe_lut__DOT__s2_addr;
    bf16_expe_lut__DOT__s2_addr = 0;
    CData/*3:0*/ bf16_expe_lut__DOT__u_decompose__DOT__status_comb;
    bf16_expe_lut__DOT__u_decompose__DOT__status_comb = 0;
    SData/*8:0*/ bf16_expe_lut__DOT__u_decompose__DOT__exponent_comb;
    bf16_expe_lut__DOT__u_decompose__DOT__exponent_comb = 0;
    CData/*0:0*/ bf16_expe_lut__DOT__u_decompose__DOT__hidden_bit_comb;
    bf16_expe_lut__DOT__u_decompose__DOT__hidden_bit_comb = 0;
    CData/*1:0*/ bf16_expe_lut__DOT__u_early_out__DOT__eo_comb;
    bf16_expe_lut__DOT__u_early_out__DOT__eo_comb = 0;
    // Body
    vlSelf->m_axis_tvalid = vlSelf->s_axis_tvalid;
    vlSelf->s_axis_tready = vlSelf->m_axis_tready;
    bf16_expe_lut__DOT__u_decompose__DOT__status_comb 
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
    bf16_expe_lut__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(bf16_expe_lut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(bf16_expe_lut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    bf16_expe_lut__DOT__u_decompose__DOT__exponent_comb 
        = ((IData)((0U != (0xdU & (IData)(bf16_expe_lut__DOT__u_decompose__DOT__status_comb))))
            ? 0U : ((2U & (IData)(bf16_expe_lut__DOT__u_decompose__DOT__status_comb))
                     ? 0x182U : (0x1ffU & ((0xffU & 
                                            ((IData)(vlSelf->s_axis_tdata) 
                                             >> 7U)) 
                                           - (IData)(0x7fU)))));
    bf16_expe_lut__DOT__s2_addr = ((0xf80U & (((IData)(bf16_expe_lut__DOT__u_decompose__DOT__exponent_comb) 
                                               - (IData)(0x17U)) 
                                              << 7U)) 
                                   | (0x7fU & (IData)(vlSelf->s_axis_tdata)));
    bf16_expe_lut__DOT__s1_decomposed = ((0x200000U 
                                          & ((IData)(vlSelf->s_axis_tdata) 
                                             << 6U)) 
                                         | (((IData)(bf16_expe_lut__DOT__u_decompose__DOT__exponent_comb) 
                                             << 0xcU) 
                                            | ((0xfe0U 
                                                & ((IData)(vlSelf->s_axis_tdata) 
                                                   << 5U)) 
                                               | (((IData)(bf16_expe_lut__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(bf16_expe_lut__DOT__u_decompose__DOT__status_comb)))));
    bf16_expe_lut__DOT__u_early_out__DOT__eo_comb = 
        ((8U & bf16_expe_lut__DOT__s1_decomposed) ? 3U
          : ((1U & bf16_expe_lut__DOT__s1_decomposed)
              ? 1U : ((4U & bf16_expe_lut__DOT__s1_decomposed)
                       ? ((0x200000U & bf16_expe_lut__DOT__s1_decomposed)
                           ? 2U : 1U) : ((0x200000U 
                                          & bf16_expe_lut__DOT__s1_decomposed)
                                          ? (VL_GTS_III(9, 0x1f7U, 
                                                        (0x1ffU 
                                                         & (bf16_expe_lut__DOT__s1_decomposed 
                                                            >> 0xcU)))
                                              ? 1U : 
                                             (VL_LTS_III(9, 7U, 
                                                         (0x1ffU 
                                                          & (bf16_expe_lut__DOT__s1_decomposed 
                                                             >> 0xcU)))
                                               ? 2U
                                               : 0U))
                                          : 1U))));
    vlSelf->m_axis_tdata = ((3U == (IData)(bf16_expe_lut__DOT__u_early_out__DOT__eo_comb))
                             ? 0xffc0U : ((1U == (IData)(bf16_expe_lut__DOT__u_early_out__DOT__eo_comb))
                                           ? 0x3f80U
                                           : ((2U == (IData)(bf16_expe_lut__DOT__u_early_out__DOT__eo_comb))
                                               ? 0U
                                               : ((0x87fU 
                                                   >= (IData)(bf16_expe_lut__DOT__s2_addr))
                                                   ? 
                                                  vlSelf->bf16_expe_lut__DOT__u_rom__DOT__rom
                                                  [bf16_expe_lut__DOT__s2_addr]
                                                   : 0U))));
}

void Vbf16_expe_lut___024root___eval_ico(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vbf16_expe_lut___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vbf16_expe_lut___024root___eval_triggers__ico(Vbf16_expe_lut___024root* vlSelf);

bool Vbf16_expe_lut___024root___eval_phase__ico(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vbf16_expe_lut___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vbf16_expe_lut___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vbf16_expe_lut___024root___eval_act(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___eval_act\n"); );
}

void Vbf16_expe_lut___024root___eval_nba(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___eval_nba\n"); );
}

void Vbf16_expe_lut___024root___eval_triggers__act(Vbf16_expe_lut___024root* vlSelf);

bool Vbf16_expe_lut___024root___eval_phase__act(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<0> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vbf16_expe_lut___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vbf16_expe_lut___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vbf16_expe_lut___024root___eval_phase__nba(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vbf16_expe_lut___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_lut___024root___dump_triggers__ico(Vbf16_expe_lut___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_lut___024root___dump_triggers__nba(Vbf16_expe_lut___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_lut___024root___dump_triggers__act(Vbf16_expe_lut___024root* vlSelf);
#endif  // VL_DEBUG

void Vbf16_expe_lut___024root___eval(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___eval\n"); );
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
            Vbf16_expe_lut___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_lut.sv", 32, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vbf16_expe_lut___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vbf16_expe_lut___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_lut.sv", 32, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vbf16_expe_lut___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("src/rtl/bf16_expe_lut.sv", 32, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vbf16_expe_lut___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vbf16_expe_lut___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vbf16_expe_lut___024root___eval_debug_assertions(Vbf16_expe_lut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_lut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_lut___024root___eval_debug_assertions\n"); );
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
