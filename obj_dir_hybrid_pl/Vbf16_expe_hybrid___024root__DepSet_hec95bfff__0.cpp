// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_hybrid.h for the primary calling header

#include "Vbf16_expe_hybrid__pch.h"
#include "Vbf16_expe_hybrid___024root.h"

VL_INLINE_OPT void Vbf16_expe_hybrid___024root___ico_sequent__TOP__0(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
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
    vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
}

void Vbf16_expe_hybrid___024root___eval_ico(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vbf16_expe_hybrid___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vbf16_expe_hybrid___024root___eval_triggers__ico(Vbf16_expe_hybrid___024root* vlSelf);

bool Vbf16_expe_hybrid___024root___eval_phase__ico(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vbf16_expe_hybrid___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vbf16_expe_hybrid___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vbf16_expe_hybrid___024root___eval_act(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_act\n"); );
}

extern const VlUnpacked<SData/*15:0*/, 1024> Vbf16_expe_hybrid__ConstPool__TABLE_hf7ff168e_0;

VL_INLINE_OPT void Vbf16_expe_hybrid___024root___nba_sequent__TOP__0(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___nba_sequent__TOP__0\n"); );
    // Init
    SData/*9:0*/ __Vtableidx1;
    __Vtableidx1 = 0;
    // Body
    if (vlSelf->rst_n) {
        if (vlSelf->s_axis_tready) {
            vlSelf->bf16_expe_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr 
                = ((0xeU & ((IData)(vlSelf->bf16_expe_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr) 
                            << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            vlSelf->bf16_expe_hybrid__DOT__s2_mantissa 
                = (0x7fU & (vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                            >> 5U));
            vlSelf->bf16_expe_hybrid__DOT__s2_sparse_exp_index 
                = (7U & ((IData)(1U) + (vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                                        >> 0xcU)));
            vlSelf->m_axis_tdata = ((3U == (IData)(vlSelf->bf16_expe_hybrid__DOT__s3_eo_code))
                                     ? 0xffc0U : ((1U 
                                                   == (IData)(vlSelf->bf16_expe_hybrid__DOT__s3_eo_code))
                                                   ? 0x3f80U
                                                   : 
                                                  ((2U 
                                                    == (IData)(vlSelf->bf16_expe_hybrid__DOT__s3_eo_code))
                                                    ? 0U
                                                    : 
                                                   ((0U 
                                                     == (IData)(vlSelf->bf16_expe_hybrid__DOT__s3_route))
                                                     ? (IData)(vlSelf->bf16_expe_hybrid__DOT__s3_sparse_data)
                                                     : 
                                                    ((1U 
                                                      == (IData)(vlSelf->bf16_expe_hybrid__DOT__s3_route))
                                                      ? (IData)(vlSelf->bf16_expe_hybrid__DOT__s3_dense_data)
                                                      : 0U)))));
            vlSelf->bf16_expe_hybrid__DOT__s3_sparse_data 
                = vlSelf->bf16_expe_hybrid__DOT__u_sparse__DOT__data_comb;
            vlSelf->bf16_expe_hybrid__DOT__s3_eo_code 
                = vlSelf->bf16_expe_hybrid__DOT__s2_eo_code;
            vlSelf->bf16_expe_hybrid__DOT__s3_route 
                = vlSelf->bf16_expe_hybrid__DOT__s2_route;
            vlSelf->bf16_expe_hybrid__DOT__s2_eo_code 
                = ((8U & vlSelf->bf16_expe_hybrid__DOT__s1_decomposed)
                    ? 3U : ((1U & vlSelf->bf16_expe_hybrid__DOT__s1_decomposed)
                             ? 1U : ((4U & vlSelf->bf16_expe_hybrid__DOT__s1_decomposed)
                                      ? ((0x200000U 
                                          & vlSelf->bf16_expe_hybrid__DOT__s1_decomposed)
                                          ? 2U : 1U)
                                      : ((0x200000U 
                                          & vlSelf->bf16_expe_hybrid__DOT__s1_decomposed)
                                          ? (VL_GTS_III(9, 0x1f7U, 
                                                        (0x1ffU 
                                                         & (vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                                                            >> 0xcU)))
                                              ? 1U : 
                                             (VL_LTS_III(9, 7U, 
                                                         (0x1ffU 
                                                          & (vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                                                             >> 0xcU)))
                                               ? 2U
                                               : 0U))
                                          : 1U))));
            vlSelf->bf16_expe_hybrid__DOT__s2_route 
                = (VL_GTES_III(9, 0x1feU, (0x1ffU & 
                                           (vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                                            >> 0xcU)))
                    ? 0U : (VL_GTES_III(9, 6U, (0x1ffU 
                                                & (vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                                                   >> 0xcU)))
                             ? 1U : 2U));
        }
    } else {
        vlSelf->bf16_expe_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr = 0U;
        vlSelf->bf16_expe_hybrid__DOT__s2_mantissa = 0U;
        vlSelf->bf16_expe_hybrid__DOT__s2_sparse_exp_index = 0U;
        vlSelf->m_axis_tdata = 0U;
        vlSelf->bf16_expe_hybrid__DOT__s3_sparse_data = 0x3f80U;
        vlSelf->bf16_expe_hybrid__DOT__s3_eo_code = 1U;
        vlSelf->bf16_expe_hybrid__DOT__s3_route = 0U;
        vlSelf->bf16_expe_hybrid__DOT__s2_eo_code = 1U;
        vlSelf->bf16_expe_hybrid__DOT__s2_route = 0U;
    }
    vlSelf->m_axis_tvalid = (1U & ((IData)(vlSelf->bf16_expe_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                   >> 3U));
    __Vtableidx1 = (((IData)(vlSelf->bf16_expe_hybrid__DOT__s2_mantissa) 
                     << 3U) | (IData)(vlSelf->bf16_expe_hybrid__DOT__s2_sparse_exp_index));
    vlSelf->bf16_expe_hybrid__DOT__u_sparse__DOT__data_comb 
        = Vbf16_expe_hybrid__ConstPool__TABLE_hf7ff168e_0
        [__Vtableidx1];
}

VL_INLINE_OPT void Vbf16_expe_hybrid___024root___nba_sequent__TOP__1(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___nba_sequent__TOP__1\n"); );
    // Body
    if (vlSelf->s_axis_tready) {
        vlSelf->bf16_expe_hybrid__DOT__s3_dense_data 
            = vlSelf->bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom
            [vlSelf->bf16_expe_hybrid__DOT__s2_dense_addr];
    }
}

VL_INLINE_OPT void Vbf16_expe_hybrid___024root___nba_sequent__TOP__2(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___nba_sequent__TOP__2\n"); );
    // Body
    if (vlSelf->rst_n) {
        if (vlSelf->s_axis_tready) {
            vlSelf->bf16_expe_hybrid__DOT__s2_dense_addr 
                = ((0x380U & (((IData)(1U) + (vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                                              >> 0xcU)) 
                              << 7U)) | (0x7fU & (vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                                                  >> 5U)));
            vlSelf->bf16_expe_hybrid__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb))
                                                   ? 0x182U
                                                   : 
                                                  (0x1ffU 
                                                   & ((0xffU 
                                                       & ((IData)(vlSelf->s_axis_tdata) 
                                                          >> 7U)) 
                                                      - (IData)(0x7fU))))) 
                                             << 0xcU) 
                                            | ((0xfe0U 
                                                & ((IData)(vlSelf->s_axis_tdata) 
                                                   << 5U)) 
                                               | (((IData)(vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb)))));
        }
    } else {
        vlSelf->bf16_expe_hybrid__DOT__s2_dense_addr = 0U;
        vlSelf->bf16_expe_hybrid__DOT__s1_decomposed = 0U;
    }
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
}

void Vbf16_expe_hybrid___024root___eval_nba(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_hybrid___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_hybrid___024root___nba_sequent__TOP__1(vlSelf);
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_hybrid___024root___nba_sequent__TOP__2(vlSelf);
    }
}

void Vbf16_expe_hybrid___024root___eval_triggers__act(Vbf16_expe_hybrid___024root* vlSelf);

bool Vbf16_expe_hybrid___024root___eval_phase__act(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<2> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vbf16_expe_hybrid___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vbf16_expe_hybrid___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vbf16_expe_hybrid___024root___eval_phase__nba(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vbf16_expe_hybrid___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_hybrid___024root___dump_triggers__ico(Vbf16_expe_hybrid___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_hybrid___024root___dump_triggers__nba(Vbf16_expe_hybrid___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_hybrid___024root___dump_triggers__act(Vbf16_expe_hybrid___024root* vlSelf);
#endif  // VL_DEBUG

void Vbf16_expe_hybrid___024root___eval(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval\n"); );
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
            Vbf16_expe_hybrid___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_hybrid.sv", 12, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vbf16_expe_hybrid___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vbf16_expe_hybrid___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_hybrid.sv", 12, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vbf16_expe_hybrid___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("src/rtl/bf16_expe_hybrid.sv", 12, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vbf16_expe_hybrid___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vbf16_expe_hybrid___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vbf16_expe_hybrid___024root___eval_debug_assertions(Vbf16_expe_hybrid___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_hybrid__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_hybrid___024root___eval_debug_assertions\n"); );
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
