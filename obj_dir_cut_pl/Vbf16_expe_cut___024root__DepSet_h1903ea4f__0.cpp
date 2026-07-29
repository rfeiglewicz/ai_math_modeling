// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_cut.h for the primary calling header

#include "Vbf16_expe_cut__pch.h"
#include "Vbf16_expe_cut___024root.h"

VL_INLINE_OPT void Vbf16_expe_cut___024root___ico_sequent__TOP__0(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
    vlSelf->bf16_expe_cut__DOT__u_decompose__DOT__status_comb 
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
    vlSelf->bf16_expe_cut__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_cut__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_cut__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
}

void Vbf16_expe_cut___024root___eval_ico(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vbf16_expe_cut___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vbf16_expe_cut___024root___eval_triggers__ico(Vbf16_expe_cut___024root* vlSelf);

bool Vbf16_expe_cut___024root___eval_phase__ico(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vbf16_expe_cut___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vbf16_expe_cut___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vbf16_expe_cut___024root___eval_act(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vbf16_expe_cut___024root___nba_sequent__TOP__0(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___nba_sequent__TOP__0\n"); );
    // Init
    SData/*15:0*/ __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 0;
    // Body
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0U;
    __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0U;
    if (vlSelf->rst_n) {
        if (vlSelf->s_axis_tready) {
            vlSelf->bf16_expe_cut__DOT__gen_axi_pipelined__DOT__valid_sr 
                = ((0xffeU & ((IData)(vlSelf->bf16_expe_cut__DOT__gen_axi_pipelined__DOT__valid_sr) 
                              << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 
                = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 1U;
            __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 
                = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 1U;
            __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 
                = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 1U;
            __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 
                = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 1U;
            __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 
                = vlSelf->bf16_expe_cut__DOT__core_data;
            __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 1U;
            vlSelf->bf16_expe_cut__DOT__s5_base_count 
                = vlSelf->bf16_expe_cut__DOT__s4_base_count;
            vlSelf->bf16_expe_cut__DOT__fe_shift = 
                (0x1fU & ((IData)(0xeU) - (vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                                           >> 0xcU)));
            vlSelf->bf16_expe_cut__DOT__fe_prod = ((IData)(0xb8aa3bU) 
                                                   * 
                                                   (0x80U 
                                                    | (0x7fU 
                                                       & (vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                                                          >> 5U))));
            vlSelf->bf16_expe_cut__DOT__s5_below_cut 
                = ((~ (IData)(vlSelf->bf16_expe_cut__DOT__s4_cut_sentinel)) 
                   & VL_LTES_III(18, (IData)(vlSelf->bf16_expe_cut__DOT__s4_frac), 
                                 (0x3ffffU & ((IData)(0x3eb02U) 
                                              + (VL_SHIFTL_III(18,18,32, 
                                                               VL_EXTENDS_II(18,9, 
                                                                             (0x1ffU 
                                                                              & ((IData)(0x7fU) 
                                                                                - (IData)(vlSelf->bf16_expe_cut__DOT__s4_base_count)))), 9U) 
                                                 + (IData)(vlSelf->bf16_expe_cut__DOT__s4_cut_dev))))));
            vlSelf->bf16_expe_cut__DOT__core_data = 
                ((3U == (IData)(vlSelf->bf16_expe_cut__DOT__s5_eo_code))
                  ? 0xffc0U : ((1U == (IData)(vlSelf->bf16_expe_cut__DOT__s5_eo_code))
                                ? 0x3f80U : ((2U == (IData)(vlSelf->bf16_expe_cut__DOT__s5_eo_code))
                                              ? 0U : 
                                             ((0U == (IData)(vlSelf->bf16_expe_cut__DOT__s5_route))
                                               ? ((0x7f80U 
                                                   & ((((IData)(0x7eU) 
                                                        - (IData)(vlSelf->bf16_expe_cut__DOT__s5_int_part)) 
                                                       + 
                                                       (1U 
                                                        & ((IData)(vlSelf->bf16_expe_cut__DOT__ladder_count) 
                                                           >> 7U))) 
                                                      << 7U)) 
                                                  | (0x7fU 
                                                     & (IData)(vlSelf->bf16_expe_cut__DOT__ladder_count)))
                                               : ((1U 
                                                   == (IData)(vlSelf->bf16_expe_cut__DOT__s5_route))
                                                   ? (IData)(vlSelf->bf16_expe_cut__DOT__s5_tail_data)
                                                   : 0U)))));
            vlSelf->bf16_expe_cut__DOT__s4_cut_sentinel 
                = VL_LTES_III(9, 0x80U, (IData)(vlSelf->bf16_expe_cut__DOT__base_count));
            vlSelf->bf16_expe_cut__DOT__s4_base_count 
                = vlSelf->bf16_expe_cut__DOT__base_count;
            vlSelf->bf16_expe_cut__DOT__s4_cut_dev 
                = vlSelf->bf16_expe_cut__DOT__u_cut_rom__DOT__rom
                [(0x7fU & (IData)(vlSelf->bf16_expe_cut__DOT__base_count))];
            vlSelf->bf16_expe_cut__DOT__s4_frac = vlSelf->bf16_expe_cut__DOT__s3_frac;
            vlSelf->bf16_expe_cut__DOT__s5_int_part 
                = vlSelf->bf16_expe_cut__DOT__s4_int_part;
            vlSelf->bf16_expe_cut__DOT__s5_tail_data 
                = vlSelf->bf16_expe_cut__DOT__s4_tail_data;
            vlSelf->bf16_expe_cut__DOT__s5_eo_code 
                = vlSelf->bf16_expe_cut__DOT__s4_eo_code;
            vlSelf->bf16_expe_cut__DOT__s5_route = vlSelf->bf16_expe_cut__DOT__s4_route;
            vlSelf->bf16_expe_cut__DOT__s3_frac = vlSelf->bf16_expe_cut__DOT__s2_frac;
            vlSelf->bf16_expe_cut__DOT__s4_int_part 
                = vlSelf->bf16_expe_cut__DOT__s3_int_part;
            vlSelf->bf16_expe_cut__DOT__s4_tail_data 
                = vlSelf->bf16_expe_cut__DOT__s3_tail_data;
            vlSelf->bf16_expe_cut__DOT__s4_eo_code 
                = vlSelf->bf16_expe_cut__DOT__s3_eo_code;
            vlSelf->bf16_expe_cut__DOT__s4_route = vlSelf->bf16_expe_cut__DOT__s3_route;
            vlSelf->bf16_expe_cut__DOT__s3_int_part 
                = vlSelf->bf16_expe_cut__DOT__s2_int_part;
            vlSelf->bf16_expe_cut__DOT__s3_eo_code 
                = vlSelf->bf16_expe_cut__DOT__fe_eo_code;
            vlSelf->bf16_expe_cut__DOT__s3_route = vlSelf->bf16_expe_cut__DOT__s2_route;
            vlSelf->bf16_expe_cut__DOT__s2_int_part 
                = (0x1ffU & (vlSelf->bf16_expe_cut__DOT____VdfgTmp_h0f01a197__0 
                             >> 0x10U));
            vlSelf->bf16_expe_cut__DOT__fe_eo_code 
                = vlSelf->bf16_expe_cut__DOT__s2_eo_code;
            vlSelf->bf16_expe_cut__DOT__s2_route = vlSelf->bf16_expe_cut__DOT__fe_route;
            vlSelf->bf16_expe_cut__DOT__s2_eo_code 
                = ((8U & vlSelf->bf16_expe_cut__DOT__s1_decomposed)
                    ? 3U : ((1U & vlSelf->bf16_expe_cut__DOT__s1_decomposed)
                             ? 1U : ((4U & vlSelf->bf16_expe_cut__DOT__s1_decomposed)
                                      ? ((0x200000U 
                                          & vlSelf->bf16_expe_cut__DOT__s1_decomposed)
                                          ? 2U : 1U)
                                      : ((0x200000U 
                                          & vlSelf->bf16_expe_cut__DOT__s1_decomposed)
                                          ? (VL_GTS_III(9, 0x1f7U, 
                                                        (0x1ffU 
                                                         & (vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                                                            >> 0xcU)))
                                              ? 1U : 
                                             (VL_LTS_III(9, 7U, 
                                                         (0x1ffU 
                                                          & (vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                                                             >> 0xcU)))
                                               ? 2U
                                               : 0U))
                                          : 1U))));
            vlSelf->bf16_expe_cut__DOT__fe_route = 
                (VL_LTS_III(9, 6U, (0x1ffU & (vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                                              >> 0xcU)))
                  ? 2U : ((IData)(((0x6000U == (0x1ff000U 
                                                & vlSelf->bf16_expe_cut__DOT__s1_decomposed)) 
                                   & (0x3aU <= (0x7fU 
                                                & (vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                                                   >> 5U)))))
                           ? 2U : ((IData)(((0x6000U 
                                             == (0x1ff000U 
                                                 & vlSelf->bf16_expe_cut__DOT__s1_decomposed)) 
                                            & (0x2fU 
                                               <= (0x7fU 
                                                   & (vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                                                      >> 5U)))))
                                    ? 1U : 0U)));
        }
    } else {
        vlSelf->bf16_expe_cut__DOT__gen_axi_pipelined__DOT__valid_sr = 0U;
        __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 1U;
        __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 1U;
        __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 1U;
        __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 1U;
        __Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 1U;
        vlSelf->bf16_expe_cut__DOT__s5_base_count = 0U;
        vlSelf->bf16_expe_cut__DOT__fe_shift = 0U;
        vlSelf->bf16_expe_cut__DOT__fe_prod = 0U;
        vlSelf->bf16_expe_cut__DOT__s5_below_cut = 0U;
        vlSelf->bf16_expe_cut__DOT__core_data = 0U;
        vlSelf->bf16_expe_cut__DOT__s4_cut_sentinel = 0U;
        vlSelf->bf16_expe_cut__DOT__s4_base_count = 0U;
        vlSelf->bf16_expe_cut__DOT__s4_cut_dev = 0U;
        vlSelf->bf16_expe_cut__DOT__s4_frac = 0U;
        vlSelf->bf16_expe_cut__DOT__s5_int_part = 0U;
        vlSelf->bf16_expe_cut__DOT__s5_tail_data = 0U;
        vlSelf->bf16_expe_cut__DOT__s5_eo_code = 1U;
        vlSelf->bf16_expe_cut__DOT__s5_route = 2U;
        vlSelf->bf16_expe_cut__DOT__s3_frac = 0U;
        vlSelf->bf16_expe_cut__DOT__s4_int_part = 0U;
        vlSelf->bf16_expe_cut__DOT__s4_tail_data = 0U;
        vlSelf->bf16_expe_cut__DOT__s4_eo_code = 1U;
        vlSelf->bf16_expe_cut__DOT__s4_route = 2U;
        vlSelf->bf16_expe_cut__DOT__s3_int_part = 0U;
        vlSelf->bf16_expe_cut__DOT__s3_eo_code = 1U;
        vlSelf->bf16_expe_cut__DOT__s3_route = 2U;
        vlSelf->bf16_expe_cut__DOT__s2_int_part = 0U;
        vlSelf->bf16_expe_cut__DOT__fe_eo_code = 1U;
        vlSelf->bf16_expe_cut__DOT__s2_route = 2U;
        vlSelf->bf16_expe_cut__DOT__s2_eo_code = 1U;
        vlSelf->bf16_expe_cut__DOT__fe_route = 2U;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0U] 
            = __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] 
            = __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] 
            = __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] 
            = __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] 
            = __Vdlyvval__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    }
    if (__Vdlyvset__bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9) {
        vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] = 0U;
    }
    vlSelf->m_axis_tvalid = (1U & ((IData)(vlSelf->bf16_expe_cut__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                   >> 0xbU));
    vlSelf->m_axis_tdata = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [4U];
    vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt 
        = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0U];
    vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt 
        = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [1U];
    vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt 
        = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [2U];
    vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt 
        = vlSelf->bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [3U];
    vlSelf->bf16_expe_cut__DOT__ladder_count = (0x1ffU 
                                                & ((IData)(vlSelf->bf16_expe_cut__DOT__s5_base_count) 
                                                   + (IData)(vlSelf->bf16_expe_cut__DOT__s5_below_cut)));
}

VL_INLINE_OPT void Vbf16_expe_cut___024root___nba_sequent__TOP__1(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___nba_sequent__TOP__1\n"); );
    // Body
    if (vlSelf->s_axis_tready) {
        vlSelf->bf16_expe_cut__DOT__s3_cand_dev = vlSelf->bf16_expe_cut__DOT__u_cand_rom__DOT__rom
            [(0xffU & ((IData)(vlSelf->bf16_expe_cut__DOT__s2_frac) 
                       >> 8U))];
        vlSelf->bf16_expe_cut__DOT__s3_tail_data = 
            ((0xaU >= (IData)(vlSelf->bf16_expe_cut__DOT__s2_tail_addr))
              ? vlSelf->bf16_expe_cut__DOT__u_tail_rom__DOT__rom
             [vlSelf->bf16_expe_cut__DOT__s2_tail_addr]
              : 0U);
    }
}

VL_INLINE_OPT void Vbf16_expe_cut___024root___nba_sequent__TOP__2(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___nba_sequent__TOP__2\n"); );
    // Body
    if (vlSelf->rst_n) {
        if (vlSelf->s_axis_tready) {
            vlSelf->bf16_expe_cut__DOT__s2_frac = (0xffffU 
                                                   & vlSelf->bf16_expe_cut__DOT____VdfgTmp_h0f01a197__0);
            vlSelf->bf16_expe_cut__DOT__s2_tail_addr 
                = vlSelf->bf16_expe_cut__DOT__fe_tail_addr;
            vlSelf->bf16_expe_cut__DOT__fe_tail_addr 
                = (0xfU & ((vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                            >> 5U) - (IData)(0xfU)));
            vlSelf->bf16_expe_cut__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_cut__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_cut__DOT__u_decompose__DOT__status_comb))
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
                                               | (((IData)(vlSelf->bf16_expe_cut__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_cut__DOT__u_decompose__DOT__status_comb)))));
        }
    } else {
        vlSelf->bf16_expe_cut__DOT__s2_frac = 0U;
        vlSelf->bf16_expe_cut__DOT__s2_tail_addr = 0U;
        vlSelf->bf16_expe_cut__DOT__fe_tail_addr = 0U;
        vlSelf->bf16_expe_cut__DOT__s1_decomposed = 0U;
    }
    vlSelf->bf16_expe_cut__DOT____VdfgTmp_h0f01a197__0 
        = (vlSelf->bf16_expe_cut__DOT__fe_prod >> (IData)(vlSelf->bf16_expe_cut__DOT__fe_shift));
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
}

VL_INLINE_OPT void Vbf16_expe_cut___024root___nba_comb__TOP__0(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___nba_comb__TOP__0\n"); );
    // Body
    vlSelf->bf16_expe_cut__DOT__base_count = (0x1ffU 
                                              & ((IData)(0x1f4U) 
                                                 + 
                                                 (((IData)(0x80U) 
                                                   - 
                                                   (0x7fU 
                                                    & ((IData)(vlSelf->bf16_expe_cut__DOT__s3_frac) 
                                                       >> 9U))) 
                                                  + (IData)(vlSelf->bf16_expe_cut__DOT__s3_cand_dev))));
}

void Vbf16_expe_cut___024root___eval_nba(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_cut___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_cut___024root___nba_sequent__TOP__1(vlSelf);
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_cut___024root___nba_sequent__TOP__2(vlSelf);
    }
    if ((3ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_cut___024root___nba_comb__TOP__0(vlSelf);
    }
}

void Vbf16_expe_cut___024root___eval_triggers__act(Vbf16_expe_cut___024root* vlSelf);

bool Vbf16_expe_cut___024root___eval_phase__act(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<2> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vbf16_expe_cut___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vbf16_expe_cut___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vbf16_expe_cut___024root___eval_phase__nba(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vbf16_expe_cut___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_cut___024root___dump_triggers__ico(Vbf16_expe_cut___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_cut___024root___dump_triggers__nba(Vbf16_expe_cut___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_cut___024root___dump_triggers__act(Vbf16_expe_cut___024root* vlSelf);
#endif  // VL_DEBUG

void Vbf16_expe_cut___024root___eval(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval\n"); );
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
            Vbf16_expe_cut___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_cut.sv", 30, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vbf16_expe_cut___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vbf16_expe_cut___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_expe_cut.sv", 30, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vbf16_expe_cut___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("src/rtl/bf16_expe_cut.sv", 30, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vbf16_expe_cut___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vbf16_expe_cut___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vbf16_expe_cut___024root___eval_debug_assertions(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_debug_assertions\n"); );
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
