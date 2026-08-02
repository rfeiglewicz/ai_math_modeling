// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_equiv_top.h for the primary calling header

#include "Vbf16_expe_equiv_top__pch.h"
#include "Vbf16_expe_equiv_top___024root.h"

VL_INLINE_OPT void Vbf16_expe_equiv_top___024root___ico_sequent__TOP__0(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___ico_sequent__TOP__0\n"); );
    // Body
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
    vlSelf->s_axis_tready = ((IData)(vlSelf->o_exp2_tready) 
                             & ((IData)(vlSelf->o_exp2opt_tready) 
                                & ((IData)(vlSelf->o_lut_tready) 
                                   & ((IData)(vlSelf->o_hybrid_tready) 
                                      & ((IData)(vlSelf->o_cut_tready) 
                                         & ((IData)(vlSelf->o_poly4_tready) 
                                            & (IData)(vlSelf->o_poly4dsp_tready)))))));
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
}

void Vbf16_expe_equiv_top___024root___eval_ico(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vbf16_expe_equiv_top___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vbf16_expe_equiv_top___024root___eval_triggers__ico(Vbf16_expe_equiv_top___024root* vlSelf);

bool Vbf16_expe_equiv_top___024root___eval_phase__ico(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vbf16_expe_equiv_top___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vbf16_expe_equiv_top___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vbf16_expe_equiv_top___024root___eval_act(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vbf16_expe_equiv_top___024root___nba_sequent__TOP__0(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___nba_sequent__TOP__0\n"); );
    // Init
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
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5 = 0;
    QData/*35:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 = 0;
    QData/*35:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    // Body
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v7 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v7 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v7 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v7 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v7 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0U;
    if (vlSelf->o_exp2opt_tready) {
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb
            [0U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb
            [1U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay
            [0U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay
            [1U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay
            [2U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay
            [0U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay
            [1U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay
            [2U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay
            [3U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay
            [4U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay
            [5U];
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5 = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__bf16_out 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__bf16_comb;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__merged_q 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_m_ext 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__sum_m_ext;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s4_int_part 
            = (0x1ffU & (- (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__unified_shifted 
                                    >> 0x26U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s7_rounded_fp 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__rounded_comb;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_shift 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__shift_9;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_is_sub 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_is_sub;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__frac_aligned 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s4_frac_part;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__res_s 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s5_unnorm_res;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s6_poly_exp 
            = (0x1ffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_s) 
                         - (IData)(0x3aU)));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_mant 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s6_norm_mant;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p2_base_exp 
            = ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_is_sub)
                ? 0x182U : (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_final_exp));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__coeff_b_stage 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__coeff_b_d;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__gen_shift_dsp__DOT__hi_prod_s 
            = ((0x7fffU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_out 
                           >> 0x10U)) * (0x1ffffU & 
                                         ((IData)(1U) 
                                          << (0x1fU 
                                              & ((IData)(9U) 
                                                 + 
                                                 vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay
                                                 [3U])))));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__gen_shift_dsp__DOT__lo_prod_s 
            = (0x1ffffffffULL & ((QData)((IData)((0xffffU 
                                                  & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_out))) 
                                 * (QData)((IData)(
                                                   (0x1ffffU 
                                                    & ((IData)(1U) 
                                                       << 
                                                       (0x1fU 
                                                        & ((IData)(9U) 
                                                           + 
                                                           vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay
                                                           [3U]))))))));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s5_unnorm_res 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__calc_res;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_s 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_comb;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s6_norm_mant 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__poly_mant_comb;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_final_exp 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__final_exponent;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__p1_is_sub 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__is_sub;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__coeff_b_d 
            = (0x1fffffU & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__packed_coeff 
                                    >> 0x15U)));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_out 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__gen_rne_reg__DOT__mant_rne_q;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__packed_coeff 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom
            [(0x7fU & ((IData)(0x7fU) - (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s4_frac_part 
                                                 >> 0x1fU))))];
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__gen_rne_reg__DOT__mant_rne_q 
            = ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__base2_d)
                ? ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__mant_src_d) 
                   << 0x16U) : (0x7fffff00U & (((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__mant_mult 
                                                 >> 8U) 
                                                + (1U 
                                                   & ((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__mant_mult 
                                                       >> 7U) 
                                                      & (IData)(
                                                                (0U 
                                                                 != 
                                                                 (0x17fU 
                                                                  & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__mant_mult)))))) 
                                               << 8U)));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s4_frac_part 
            = (0x3fffffffffULL & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__unified_shifted);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__mant_src_d 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_src;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__mant_mult 
            = (0x7fffffffU & ((IData)(0x5c551dU) * (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_src)));
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_log2e_mult__DOT__base2_d 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_base2;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_base2 = 0U;
    }
    if (vlSelf->o_poly4_tready) {
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v3 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_eo 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_eo;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_route 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_route;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_tail_addr 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_tail;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_int_part 
            = (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_aligned 
                         >> 0x11U));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__core_data;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_eo 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_eo;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_route 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_route;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_tail 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_tail;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_eo 
            = ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed)
                ? 3U : ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed)
                         ? 1U : ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed)
                                  ? ((0x200000U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed)
                                      ? 2U : 1U) : 
                                 ((0x200000U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed)
                                   ? (VL_GTS_III(9, 0x1f7U, 
                                                 (0x1ffU 
                                                  & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                                                     >> 0xcU)))
                                       ? 1U : (VL_LTS_III(9, 7U, 
                                                          (0x1ffU 
                                                           & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                                                              >> 0xcU)))
                                                ? 2U
                                                : 0U))
                                   : 1U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_route 
            = (VL_LTS_III(9, 6U, (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                                            >> 0xcU)))
                ? 2U : ((IData)(((0x6000U == (0x1ff000U 
                                              & vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed)) 
                                 & (0x3aU <= (0x7fU 
                                              & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                                                 >> 5U)))))
                         ? 2U : ((IData)(((0x6000U 
                                           == (0x1ff000U 
                                               & vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed)) 
                                          & (0x2fU 
                                             <= (0x7fU 
                                                 & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                                                    >> 5U)))))
                                  ? 1U : 0U)));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_tail 
            = (0xfU & ((vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                        >> 5U) - (IData)(0xfU)));
    }
    if (vlSelf->o_poly4dsp_tready) {
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v4 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v5 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v6 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v7 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v7 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v4 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v5 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v6 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v7 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v7 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v4 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v5 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v6 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v7 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v7 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v4 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v5 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v6 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v7 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v7 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v1 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v2 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v3 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v4 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v5 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v6 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v7 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v7 = 1U;
        __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_int_part 
            = (0x1ffU & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__scaled_comb 
                                 >> 0x20U)));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_onehot 
            = (0xffffU & ((IData)(1U) << (0xfU & ((IData)(9U) 
                                                  + 
                                                  (vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                                                   >> 0xcU)))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_t7 
            = (((IData)(0xb8aa3bU) * (0x80U | (0x7fU 
                                               & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                                                  >> 5U)))) 
               >> 7U);
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_eo 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_eo;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_route 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_route;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_tail_addr 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_tail;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head 
            = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__core_data;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_eo 
            = ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed)
                ? 3U : ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed)
                         ? 1U : ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed)
                                  ? ((0x200000U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed)
                                      ? 2U : 1U) : 
                                 ((0x200000U & vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed)
                                   ? (VL_GTS_III(9, 0x1f7U, 
                                                 (0x1ffU 
                                                  & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                                                     >> 0xcU)))
                                       ? 1U : (VL_LTS_III(9, 7U, 
                                                          (0x1ffU 
                                                           & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                                                              >> 0xcU)))
                                                ? 2U
                                                : 0U))
                                   : 1U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_route 
            = (VL_LTS_III(9, 6U, (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                                            >> 0xcU)))
                ? 2U : ((IData)(((0x6000U == (0x1ff000U 
                                              & vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed)) 
                                 & (0x3aU <= (0x7fU 
                                              & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                                                 >> 5U)))))
                         ? 2U : ((IData)(((0x6000U 
                                           == (0x1ff000U 
                                               & vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed)) 
                                          & (0x2fU 
                                             <= (0x7fU 
                                                 & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                                                    >> 5U)))))
                                  ? 1U : 0U)));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_tail 
            = (0xfU & ((vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                        >> 5U) - (IData)(0xfU)));
    }
    if (vlSelf->o_cut_tready) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_cand_dev 
            = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cand_rom__DOT__rom
            [(0xffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_frac) 
                       >> 8U))];
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[0U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v1;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[4U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v3;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[5U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[6U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay__v5;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[5U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[6U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v1;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay__v2;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg
        [0U];
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
        = (0x7ffffffffffffULL & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
                                 + VL_SHIFTL_QQI(51,51,32, 
                                                 vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg
                                                 [1U], 0xfU)));
    vlSelf->o_exp2opt_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__bf16_out;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s4_int_part;
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
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__ax_unsigned 
        = (0x7ffffffffffffffULL & VL_SHIFTL_QQI(59,59,32, vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__merged_q, 8U));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__neg_ax 
        = (0x3fffffffffffffffULL & (- VL_EXTENDS_QQ(62,60, bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__ax_unsigned)));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__b_aligned 
        = (0x3fffffffffffffffULL & ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__coeff_b_stage)) 
                                    << 0x26U));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__calc_res 
        = (0x3fffffffffffffffULL & (bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__b_aligned 
                                    + bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__neg_ax));
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
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__final_exponent 
        = (0x1ffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s6_poly_exp) 
                     + vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__int_delay
                     [6U]));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__is_sub 
        = VL_GTS_III(9, 0x182U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__final_exponent));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__shift_9 
        = ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__is_sub)
            ? (0x1ffU & ((IData)(0x33U) + ((IData)(0x182U) 
                                           - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_round__DOT__final_exponent))))
            : 0x33U);
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_comb = 0x1ffU;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0x3dU;
    {
        while (VL_LTES_III(32, 0U, bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) {
            if (((0x3dU >= (0x3fU & bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) 
                 && (1U & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s5_unnorm_res 
                                   >> (0x3fU & bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i)))))) {
                vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_comb 
                    = (0x1ffU & bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i);
                goto __Vlabel1;
            }
            bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                = (bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                   - (IData)(1U));
        }
        __Vlabel1: ;
    }
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shift_amt 
        = (0x1ffU & ((IData)(0x3dU) - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__msb_idx_s)));
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shifted_res 
        = (0x3fffffffffffffffULL & VL_SHIFTL_QQI(62,62,9, vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__res_s, (IData)(bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shift_amt)));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__poly_mant_comb 
        = (0x7ffffffffffffffULL & (bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_normalize__DOT__shifted_res 
                                   >> 3U));
    vlSelf->o_poly4dsp_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [3U];
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [1U];
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [2U];
    vlSelf->o_poly4_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [6U];
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [1U];
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [2U];
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [3U];
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [4U];
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [5U];
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[0U] 
        = (0xfffffffffULL & ((QData)((IData)((0x1fffffU 
                                              & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__packed_coeff)))) 
                             * (QData)((IData)((0x7fffU 
                                                & (IData)(
                                                          (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__frac_aligned 
                                                           >> 8U)))))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[1U] 
        = (0xfffffffffULL & ((QData)((IData)((0x1fffffU 
                                              & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__packed_coeff)))) 
                             * (QData)((IData)((0x7fffU 
                                                & (IData)(
                                                          (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_lin_approx__DOT__frac_aligned 
                                                           >> 0x17U)))))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__unified_shifted 
        = (0x7fffffffffffULL & (VL_SHIFTL_QQI(47,47,32, (QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__gen_shift_dsp__DOT__hi_prod_s)), 0x10U) 
                                | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_unified_shift__DOT__gen_shift_dsp__DOT__lo_prod_s));
}

extern const VlUnpacked<SData/*15:0*/, 1024> Vbf16_expe_equiv_top__ConstPool__TABLE_hf7ff168e_0;

VL_INLINE_OPT void Vbf16_expe_equiv_top___024root___nba_sequent__TOP__1(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___nba_sequent__TOP__1\n"); );
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
    SData/*9:0*/ __Vtableidx1;
    __Vtableidx1 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v1 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v3 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v5 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v7;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v7 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v9;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v9 = 0;
    SData/*8:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v11;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v11 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v1 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v3 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v5 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v7;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v7 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v9;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v9 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v11;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v11 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v13;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v13 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v15;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v15 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v17;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v17 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v19;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v19 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v21;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v21 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v23;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v23 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v25;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v25 = 0;
    QData/*35:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1 = 0;
    QData/*35:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v3 = 0;
    QData/*35:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v5 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v1 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v3 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v5 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v7;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v7 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v9;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v9 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v11;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v11 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v13;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v13 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v15;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v15 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v17;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v17 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v19;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v19 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v21;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v21 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v23;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v23 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v25;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v25 = 0;
    CData/*1:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 0;
    // Body
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v5 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v5 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v5 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v7 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v9 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v11 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v5 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v7 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v9 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v11 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v13 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v15 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v17 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v19 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v21 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v23 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v25 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v3 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v5 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v7 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v9 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v11 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v13 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v15 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v17 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v19 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v21 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v23 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v25 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0U;
    __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0U;
    if (vlSelf->rst_n) {
        if (vlSelf->o_lut_tready) {
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay
                [0U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0 = 1U;
            vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__gen_axi_pipelined__DOT__vld_sr 
                = ((0xfffeU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__gen_axi_pipelined__DOT__vld_sr) 
                               << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__8__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__9__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__10__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__11__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 1U;
            vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_eo_code 
                = ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed)
                    ? 3U : ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed)
                             ? 1U : ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed)
                                      ? ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed)
                                          ? 2U : 1U)
                                      : ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed)
                                          ? (VL_GTS_III(9, 0x1f7U, 
                                                        (0x1ffU 
                                                         & (vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed 
                                                            >> 0xcU)))
                                              ? 1U : 
                                             (VL_LTS_III(9, 7U, 
                                                         (0x1ffU 
                                                          & (vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed 
                                                             >> 0xcU)))
                                               ? 2U
                                               : 0U))
                                          : 1U))));
            vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head 
                = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__core_data;
            vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__core_data 
                = ((3U == vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay
                    [1U]) ? 0xffc0U : ((1U == vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay
                                        [1U]) ? 0x3f80U
                                        : ((2U == vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay
                                            [1U]) ? 0U
                                            : (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s3_rom_data))));
        }
        if (vlSelf->o_hybrid_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr 
                = ((0xfffeU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr) 
                               << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__8__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__9__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__10__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__11__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 1U;
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_mantissa 
                = (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                            >> 5U));
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_sparse_exp_index 
                = (7U & ((IData)(1U) + (vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                                        >> 0xcU)));
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__core_data;
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__core_data 
                = ((3U == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_eo_code))
                    ? 0xffc0U : ((1U == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_eo_code))
                                  ? 0x3f80U : ((2U 
                                                == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_eo_code))
                                                ? 0U
                                                : (
                                                   (0U 
                                                    == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_route))
                                                    ? (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_sparse_data)
                                                    : 
                                                   ((1U 
                                                     == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_route))
                                                     ? (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_dense_data)
                                                     : 0U)))));
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_sparse_data 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_sparse__DOT__data_comb;
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_eo_code 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_eo_code;
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_route 
                = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_route;
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_eo_code 
                = ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed)
                    ? 3U : ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed)
                             ? 1U : ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed)
                                      ? ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed)
                                          ? 2U : 1U)
                                      : ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed)
                                          ? (VL_GTS_III(9, 0x1f7U, 
                                                        (0x1ffU 
                                                         & (vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                                                            >> 0xcU)))
                                              ? 1U : 
                                             (VL_LTS_III(9, 7U, 
                                                         (0x1ffU 
                                                          & (vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                                                             >> 0xcU)))
                                               ? 2U
                                               : 0U))
                                          : 1U))));
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_route 
                = (VL_GTES_III(9, 0x1feU, (0x1ffU & 
                                           (vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                                            >> 0xcU)))
                    ? 0U : (VL_GTES_III(9, 6U, (0x1ffU 
                                                & (vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                                                   >> 0xcU)))
                             ? 1U : 2U));
        }
        if (vlSelf->o_cut_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__gen_axi_pipelined__DOT__valid_sr 
                = ((0xfffeU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__gen_axi_pipelined__DOT__valid_sr) 
                               << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__8__KET____DOT__nxt;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 1U;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_base_count 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_base_count;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_shift 
                = (0x1fU & ((IData)(0xeU) - (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                                             >> 0xcU)));
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_prod 
                = ((IData)(0xb8aa3bU) * (0x80U | (0x7fU 
                                                  & (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                                                     >> 5U))));
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_below_cut 
                = ((~ (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_cut_sentinel)) 
                   & VL_LTES_III(18, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_frac), 
                                 (0x3ffffU & ((IData)(0x3eb02U) 
                                              + (VL_SHIFTL_III(18,18,32, 
                                                               VL_EXTENDS_II(18,9, 
                                                                             (0x1ffU 
                                                                              & ((IData)(0x7fU) 
                                                                                - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_base_count)))), 9U) 
                                                 + (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_cut_dev))))));
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__core_data;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_cut_sentinel 
                = VL_LTES_III(9, 0x80U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__base_count));
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_base_count 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__base_count;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_cut_dev 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_cut_rom__DOT__rom
                [(0x7fU & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__base_count))];
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_frac 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_frac;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__core_data 
                = ((3U == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_eo_code))
                    ? 0xffc0U : ((1U == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_eo_code))
                                  ? 0x3f80U : ((2U 
                                                == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_eo_code))
                                                ? 0U
                                                : (
                                                   (0U 
                                                    == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_route))
                                                    ? 
                                                   ((0x7f80U 
                                                     & ((((IData)(0x7eU) 
                                                          - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_int_part)) 
                                                         + 
                                                         (1U 
                                                          & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__ladder_count) 
                                                             >> 7U))) 
                                                        << 7U)) 
                                                    | (0x7fU 
                                                       & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__ladder_count)))
                                                    : 
                                                   ((1U 
                                                     == (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_route))
                                                     ? (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_tail_data)
                                                     : 0U)))));
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_frac 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_frac;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_int_part 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_int_part;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_tail_data 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_tail_data;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_eo_code 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_eo_code;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_route 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_route;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_frac 
                = (0xffffU & vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT____VdfgTmp_h0f01a197__0);
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_int_part 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_int_part;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_tail_data 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_tail_data;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_eo_code 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_eo_code;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_route 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_route;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_int_part 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_int_part;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_eo_code 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_eo_code;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_route 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_route;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_int_part 
                = (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT____VdfgTmp_h0f01a197__0 
                             >> 0x10U));
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_eo_code 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_eo_code;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_route 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_route;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_eo_code 
                = ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed)
                    ? 3U : ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed)
                             ? 1U : ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed)
                                      ? ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed)
                                          ? 2U : 1U)
                                      : ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed)
                                          ? (VL_GTS_III(9, 0x1f7U, 
                                                        (0x1ffU 
                                                         & (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                                                            >> 0xcU)))
                                              ? 1U : 
                                             (VL_LTS_III(9, 7U, 
                                                         (0x1ffU 
                                                          & (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                                                             >> 0xcU)))
                                               ? 2U
                                               : 0U))
                                          : 1U))));
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_route 
                = (VL_LTS_III(9, 6U, (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                                                >> 0xcU)))
                    ? 2U : ((IData)(((0x6000U == (0x1ff000U 
                                                  & vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed)) 
                                     & (0x3aU <= (0x7fU 
                                                  & (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                                                     >> 5U)))))
                             ? 2U : ((IData)(((0x6000U 
                                               == (0x1ff000U 
                                                   & vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed)) 
                                              & (0x2fU 
                                                 <= 
                                                 (0x7fU 
                                                  & (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                                                     >> 5U)))))
                                      ? 1U : 0U)));
        }
        if (vlSelf->o_poly4_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_axi_pipelined__DOT__valid_sr 
                = ((0xfffeU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_axi_pipelined__DOT__valid_sr) 
                               << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__core_data 
                = ((3U == vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo
                    [4U]) ? 0xffc0U : ((1U == vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo
                                        [4U]) ? 0x3f80U
                                        : ((2U == vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo
                                            [4U]) ? 0U
                                            : ((0U 
                                                == 
                                                vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route
                                                [4U])
                                                ? (
                                                   (0x7f80U 
                                                    & ((((IData)(0x7eU) 
                                                         - 
                                                         vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int
                                                         [4U]) 
                                                        + 
                                                        VL_SHIFTR_III(8,8,32, 
                                                                      (0xffU 
                                                                       & VL_SHIFTRS_III(25,25,32, vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out, 0xfU)), 7U)) 
                                                       << 7U)) 
                                                   | (0x7fU 
                                                      & VL_SHIFTRS_III(25,25,32, vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out, 0xfU)))
                                                : (
                                                   (1U 
                                                    == 
                                                    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route
                                                    [4U])
                                                    ? 
                                                   vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail
                                                   [4U]
                                                    : 0U)))));
        }
        if (vlSelf->o_exp2opt_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__gen_axi_pipelined__DOT__vld_sr 
                = ((0xfffeU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__gen_axi_pipelined__DOT__vld_sr) 
                               << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [0U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [1U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [2U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [3U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [4U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [5U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [6U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [7U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [8U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [9U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [0xaU];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [0xbU];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [0xcU];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24 = 1U;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s2_eo_code 
                = ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed)
                    ? 3U : ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed)
                             ? 1U : ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed)
                                      ? ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed)
                                          ? 2U : 1U)
                                      : ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed)
                                          ? (VL_GTS_III(9, 0x1f7U, 
                                                        (0x1ffU 
                                                         & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
                                                            >> 0xcU)))
                                              ? 1U : 
                                             (VL_LTS_III(9, 7U, 
                                                         (0x1ffU 
                                                          & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
                                                             >> 0xcU)))
                                               ? 2U
                                               : 0U))
                                          : 1U))));
        }
        if (vlSelf->o_exp2_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__gen_axi_pipelined__DOT__vld_sr 
                = ((0xfffeU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__gen_axi_pipelined__DOT__vld_sr) 
                               << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                [0U];
            vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v0 = 1U;
            vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v2 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                [1U];
            vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v2 = 1U;
            vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v4 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                [2U];
            vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v4 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb
                [0U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb
                [1U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb
                [2U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay
                [0U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay
                [1U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay
                [2U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay
                [3U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay
                [4U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay
                [5U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [0U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [1U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [2U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [3U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [4U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [5U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [6U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [7U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [8U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [9U];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [0xaU];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [0xbU];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22 = 1U;
            __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [0xcU];
            __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24 = 1U;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__bf16_out 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_recompose__DOT__bf16_comb;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__merged_q 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_m_ext 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__sum_m_ext;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s7_rounded_fp 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__rounded_comb;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__shift_9;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_is_sub 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_is_sub;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__res_s 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s5_unnorm_res;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_poly_exp 
                = (0x1ffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_s) 
                             - (IData)(0x3aU)));
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_mant 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_norm_mant;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s2_eo_code 
                = ((8U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed)
                    ? 3U : ((1U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed)
                             ? 1U : ((4U & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed)
                                      ? ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed)
                                          ? 2U : 1U)
                                      : ((0x200000U 
                                          & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed)
                                          ? (VL_GTS_III(9, 0x1f7U, 
                                                        (0x1ffU 
                                                         & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                                                            >> 0xcU)))
                                              ? 1U : 
                                             (VL_LTS_III(9, 7U, 
                                                         (0x1ffU 
                                                          & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                                                             >> 0xcU)))
                                               ? 2U
                                               : 0U))
                                          : 1U))));
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_base_exp 
                = ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_is_sub)
                    ? 0x182U : (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_final_exp));
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_int_part 
                = (0x1ffU & (- (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_unified_shift__DOT__unified_shifted 
                                        >> 0x26U))));
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__frac_aligned 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_frac_part;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__coeff_b_stage 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__coeff_b_d;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s5_unnorm_res 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__calc_res;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_s 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_comb;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_norm_mant 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__poly_mant_comb;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))
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
                                               | (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))));
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_final_exp 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__final_exponent;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_is_sub 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__is_sub;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__coeff_b_d 
                = (0x1fffffU & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__packed_coeff 
                                        >> 0x15U)));
        }
        if (vlSelf->o_poly4dsp_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_axi_pipelined__DOT__valid_sr 
                = ((0xfffeU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_axi_pipelined__DOT__valid_sr) 
                               << 1U)) | (IData)(vlSelf->s_axis_tvalid));
            vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__core_data 
                = ((3U == vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
                    [8U]) ? 0xffc0U : ((1U == vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
                                        [8U]) ? 0x3f80U
                                        : ((2U == vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo
                                            [8U]) ? 0U
                                            : ((0U 
                                                == 
                                                vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
                                                [8U])
                                                ? (
                                                   (0x7f80U 
                                                    & ((((IData)(0x7eU) 
                                                         - 
                                                         vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int
                                                         [8U]) 
                                                        + 
                                                        VL_SHIFTR_III(8,8,32, 
                                                                      (0xffU 
                                                                       & VL_SHIFTRS_III(25,25,32, vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out, 0xfU)), 7U)) 
                                                       << 7U)) 
                                                   | (0x7fU 
                                                      & VL_SHIFTRS_III(25,25,32, vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out, 0xfU)))
                                                : (
                                                   (1U 
                                                    == 
                                                    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route
                                                    [8U])
                                                    ? 
                                                   vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail
                                                   [8U]
                                                    : 0U)))));
            vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))
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
                                               | (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))));
        }
    } else {
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v1 = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__gen_axi_pipelined__DOT__vld_sr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__gen_axi_pipelined__DOT__valid_sr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_axi_pipelined__DOT__valid_sr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__gen_axi_pipelined__DOT__vld_sr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__gen_axi_pipelined__DOT__vld_sr = 0U;
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v1 = 1U;
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v3 = 1U;
        vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v5 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v3 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v5 = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_axi_pipelined__DOT__valid_sr = 0U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v1 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v3 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v5 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v7 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v9 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v11 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v1 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v3 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v5 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v7 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v9 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v11 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v13 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v15 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v17 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v19 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v21 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v23 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v25 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v1 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v3 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v5 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v7 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v9 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v11 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v13 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v15 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v17 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v19 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v21 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v23 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v25 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 1U;
        __Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__bf16_out = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__merged_q = 0ULL;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_m_ext = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s7_rounded_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_shift = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s2_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_mantissa = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_sparse_exp_index = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_base_count = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_shift = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_prod = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_is_sub = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__res_s = 0ULL;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_poly_exp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_mant = 0ULL;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s2_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p2_base_exp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_below_cut = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_int_part = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__frac_aligned = 0ULL;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__coeff_b_stage = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__core_data = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__core_data = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s1_decomposed = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s5_unnorm_res = 0ULL;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_s = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_norm_mant = 0ULL;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_final_exp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__p1_is_sub = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_cut_sentinel = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_base_count = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_cut_dev = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_frac = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__coeff_b_d = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__core_data = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__core_data = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__core_data = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_frac = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_sparse_data = 0x3f80U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_route = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_int_part = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_tail_data = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_route = 2U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_frac = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_route = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_int_part = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_tail_data = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s4_route = 2U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_int_part = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_route = 2U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_int_part = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_route = 2U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_eo_code = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_route = 2U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[0U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[0U] = 0ULL;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[1U] = 0ULL;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg[2U] = 0ULL;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[1U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[2U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[3U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[4U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v6;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[4U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[5U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v8;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v9) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[5U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[6U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v10;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay__v11) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[6U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[1U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[2U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[3U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[4U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v6;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[4U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[5U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v8;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v9) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[5U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[6U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v10;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v11) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[6U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[7U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v12;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v13) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[7U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[8U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v14;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v15) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[8U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[9U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v16;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v17) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[9U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0xaU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v18;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v19) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0xaU] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0xbU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v20;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v21) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0xbU] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0xcU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v22;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v23) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0xcU] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0xdU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v24;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay__v25) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0xdU] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[1U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[2U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[3U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[4U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v6;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[4U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[5U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v8;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v9) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[5U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[6U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v10;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v11) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[6U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[7U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v12;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v13) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[7U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[8U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v14;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v15) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[8U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[9U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v16;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v17) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[9U] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0xaU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v18;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v19) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0xaU] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0xbU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v20;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v21) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0xbU] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0xcU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v22;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v23) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0xcU] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0xdU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v24;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay__v25) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0xdU] = 1U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[5U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[5U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[6U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[6U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[7U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[7U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[8U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[8U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[9U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[9U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0xaU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0xaU] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0xbU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0xbU] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[5U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[5U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[6U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[6U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[7U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[7U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[8U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[8U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[9U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v16;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v17) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[9U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0xaU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v18;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v19) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0xaU] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0xbU] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v20;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v21) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0xbU] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v4;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[3U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v6;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[4U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[5U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v8;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v9) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[5U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[6U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v10;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v11) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[6U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[7U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v12;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v13) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[7U] = 0U;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[8U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v14;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v15) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[8U] = 0U;
    }
    vlSelf->o_lut_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__gen_axi_pipelined__DOT__vld_sr) 
                                  >> 0xfU));
    vlSelf->o_hybrid_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                     >> 0xfU));
    vlSelf->o_cut_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                  >> 0xfU));
    vlSelf->o_poly4_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                    >> 0xfU));
    vlSelf->o_exp2opt_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__gen_axi_pipelined__DOT__vld_sr) 
                                      >> 0xfU));
    vlSelf->o_exp2_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__gen_axi_pipelined__DOT__vld_sr) 
                                   >> 0xfU));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg
        [0U];
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
        = (0x7ffffffffffffffULL & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
                                   + VL_SHIFTL_QQI(59,59,32, 
                                                   vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg
                                                   [1U], 0xfU)));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
        = (0x7ffffffffffffffULL & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__merged 
                                   + VL_SHIFTL_QQI(59,59,32, 
                                                   vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_reg
                                                   [2U], 0x1eU)));
    vlSelf->o_poly4dsp_tvalid = (1U & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                       >> 0xfU));
    vlSelf->o_exp2_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__bf16_out;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s2_eo_code;
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
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s2_eo_code;
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
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_int_part;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__ax_unsigned 
        = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__gen_merge_reg__DOT__merged_q;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__neg_ax 
        = (0x3fffffffffffffffULL & (- VL_EXTENDS_QQ(62,60, bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__ax_unsigned)));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__b_aligned 
        = (0x3fffffffffffffffULL & ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__coeff_b_stage)) 
                                    << 0x26U));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__calc_res 
        = (0x3fffffffffffffffULL & (bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__b_aligned 
                                    + bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__neg_ax));
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
    if ((3U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
         [0xdU])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (8U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (0x200000U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (0x800U | (0x3ff01fU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp));
    } else if ((1U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [0xdU])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (0x200fffU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (0x10U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
    } else if ((2U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__eo_delay
                [0xdU])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = (1U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp);
    } else {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s8_final_fp 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s7_rounded_fp;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__final_exponent 
        = (0x1ffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s6_poly_exp) 
                     + vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__int_delay
                     [6U]));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__is_sub 
        = VL_GTS_III(9, 0x182U, (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__final_exponent));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__shift_9 
        = ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__is_sub)
            ? (0x1ffU & ((IData)(0x33U) + ((IData)(0x182U) 
                                           - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_round__DOT__final_exponent))))
            : 0x33U);
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_comb = 0x1ffU;
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0x3dU;
    {
        while (VL_LTES_III(32, 0U, bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) {
            if (((0x3dU >= (0x3fU & bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) 
                 && (1U & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s5_unnorm_res 
                                   >> (0x3fU & bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)))))) {
                vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_comb 
                    = (0x1ffU & bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i);
                goto __Vlabel2;
            }
            bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                = (bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                   - (IData)(1U));
        }
        __Vlabel2: ;
    }
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shift_amt 
        = (0x1ffU & ((IData)(0x3dU) - (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__msb_idx_s)));
    bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shifted_res 
        = (0x3fffffffffffffffULL & VL_SHIFTL_QQI(62,62,9, vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__res_s, (IData)(bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shift_amt)));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__poly_mant_comb 
        = (0x7ffffffffffffffULL & (bf16_expe_equiv_top__DOT__u_exp2__DOT__u_normalize__DOT__shifted_res 
                                   >> 3U));
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
    vlSelf->o_lut_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0xbU];
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
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__8__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [7U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__9__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [8U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__10__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [9U];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__11__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0xaU];
    vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_eo_code;
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay[1U] 
            = __Vdlyvval__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v0;
    }
    if (__Vdlyvset__bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__eo_delay[1U] = 1U;
    }
    vlSelf->o_hybrid_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0xbU];
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
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__8__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [7U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__9__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [8U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__10__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [9U];
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__11__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0xaU];
    vlSelf->o_cut_tdata = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [8U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [1U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [2U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [3U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [4U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [5U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [6U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__8__KET____DOT__nxt 
        = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [7U];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__ladder_count 
        = (0x1ffU & ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_base_count) 
                     + (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s5_below_cut)));
    __Vtableidx1 = (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_mantissa) 
                     << 3U) | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_sparse_exp_index));
    vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_sparse__DOT__data_comb 
        = Vbf16_expe_equiv_top__ConstPool__TABLE_hf7ff168e_0
        [__Vtableidx1];
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT____VdfgTmp_h0f01a197__0 
        = (vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_prod 
           >> (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_shift));
}

VL_INLINE_OPT void Vbf16_expe_equiv_top___024root___nba_sequent__TOP__2(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___nba_sequent__TOP__2\n"); );
    // Body
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_eo;
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_eo__v7;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_route;
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_route__v7;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_int_part;
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_int__v7;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[0U] 
        = ((0xaU >= (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_tail_addr))
            ? vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__u_tail_rom__DOT__rom
           [vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_tail_addr]
            : 0U);
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_tail__v7;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_eo;
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_eo__v3;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_route;
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_route__v3;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_int_part;
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_int__v3;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail[0U] 
        = ((0xaU >= (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_tail_addr))
            ? vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_tail_rom__DOT__rom
           [vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_tail_addr]
            : 0U);
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_tail__v3;
    }
    if (vlSelf->o_exp2opt_tready) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
            = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                             << 6U)) | ((((IData)((0U 
                                                   != 
                                                   (0xdU 
                                                    & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))))
                                           ? 0U : (
                                                   (2U 
                                                    & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))
                                                    ? 0x182U
                                                    : 
                                                   (0x1ffU 
                                                    & ((0xffU 
                                                        & ((IData)(vlSelf->s_axis_tdata) 
                                                           >> 7U)) 
                                                       - (IData)(0x7fU))))) 
                                         << 0xcU) | 
                                        ((0xfe0U & 
                                          ((IData)(vlSelf->s_axis_tdata) 
                                           << 5U)) 
                                         | (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_decompose__DOT__hidden_bit_comb) 
                                             << 4U) 
                                            | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))));
    }
    if (vlSelf->o_poly4dsp_tready) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x68080010000ULL 
                                                          + vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__3__KET____DOT__u_step__DOT__mult_r)), 0x11U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__3__KET____DOT__u_step__DOT__mult_r 
            = (0x7ffffffffffULL & VL_MULS_QQQ(43, (0x7ffffffffffULL 
                                                   & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out)), 
                                              (0x7ffffffffffULL 
                                               & VL_EXTENDS_QI(43,18, 
                                                               vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
                                                               [6U]))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x74e92010000ULL 
                                                          + vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__2__KET____DOT__u_step__DOT__mult_r)), 0x11U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__2__KET____DOT__u_step__DOT__mult_r 
            = (0x7ffffffffffULL & VL_MULS_QQQ(43, (0x7ffffffffffULL 
                                                   & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out)), 
                                              (0x7ffffffffffULL 
                                               & VL_EXTENDS_QI(43,18, 
                                                               vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
                                                               [4U]))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x3d54010000ULL 
                                                          + vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__1__KET____DOT__u_step__DOT__mult_r)), 0x11U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__1__KET____DOT__u_step__DOT__mult_r 
            = (0x7ffffffffffULL & VL_MULS_QQQ(43, (0x7ffffffffffULL 
                                                   & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out)), 
                                              (0x7ffffffffffULL 
                                               & VL_EXTENDS_QI(43,18, 
                                                               vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac
                                                               [2U]))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x7f252010000ULL 
                                                          + vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__0__KET____DOT__u_step__DOT__mult_r)), 0x11U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_horner__BRA__0__KET____DOT__u_step__DOT__mult_r 
            = (0x7ffffffffffULL & VL_MULS_QQQ(43, 0xe400ULL, 
                                              (0x7ffffffffffULL 
                                               & VL_EXTENDS_QI(43,18, vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_frac))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_frac 
            = (0x1ffffU & (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__scaled_comb 
                                   >> 0xfU)));
    }
    if (vlSelf->o_poly4_tready) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x68080010000ULL 
                                                          + 
                                                          VL_MULS_QQQ(43, 
                                                                      (0x7ffffffffffULL 
                                                                       & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out)), 
                                                                      (0x7ffffffffffULL 
                                                                       & VL_EXTENDS_QI(43,18, 
                                                                                vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac
                                                                                [3U]))))), 0x11U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x74e92010000ULL 
                                                          + 
                                                          VL_MULS_QQQ(43, 
                                                                      (0x7ffffffffffULL 
                                                                       & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out)), 
                                                                      (0x7ffffffffffULL 
                                                                       & VL_EXTENDS_QI(43,18, 
                                                                                vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac
                                                                                [2U]))))), 0x11U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x3d54010000ULL 
                                                          + 
                                                          VL_MULS_QQQ(43, 
                                                                      (0x7ffffffffffULL 
                                                                       & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out)), 
                                                                      (0x7ffffffffffULL 
                                                                       & VL_EXTENDS_QI(43,18, 
                                                                                vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac
                                                                                [1U]))))), 0x11U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x7f252010000ULL 
                                                          + 
                                                          VL_MULS_QQQ(43, 0xe400ULL, 
                                                                      (0x7ffffffffffULL 
                                                                       & VL_EXTENDS_QI(43,18, vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_frac))))), 0x11U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_frac 
            = (0x1ffffU & vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_aligned);
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_fe2_reg__DOT__fe2_aligned 
            = (0x3ffffffU & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_prod 
                             >> (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_shift)));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_prod 
            = ((IData)(0xb8aa3bU) * (0x80U | (0x7fU 
                                              & (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                                                 >> 5U))));
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__gen_frontend_barrel__DOT__fe1_shift 
            = (0x1fU & ((IData)(0xdU) - (vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                                         >> 0xcU)));
    }
    if (vlSelf->o_exp2_tready) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__packed_coeff 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom
            [(0x7fU & ((IData)(0x7fU) - (IData)((vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_frac_part 
                                                 >> 0x1fU))))];
    }
    if (vlSelf->o_lut_tready) {
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s3_rom_data 
            = ((0x87fU >= (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_addr))
                ? vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_rom__DOT__rom
               [vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_addr]
                : 0U);
    }
    if (vlSelf->o_hybrid_tready) {
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s3_dense_data 
            = vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_dense_rom__DOT__rom
            [vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_dense_addr];
    }
    if (vlSelf->o_cut_tready) {
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_tail_data 
            = ((0xaU >= (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_tail_addr))
                ? vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_tail_rom__DOT__rom
               [vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_tail_addr]
                : 0U);
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__exp_delay[0U] 
        = (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
                     >> 0xcU));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s3_mant_src 
        = ((0x80U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
                     << 3U)) | (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s1_decomposed 
                                         >> 5U)));
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac__v3;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__ctrl_frac[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s2_frac;
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v6) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v7) {
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac__v7;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__scaled_comb 
        = (0x1ffffffffffULL & ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_t7)) 
                               * (QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__gen_frontend_dsp__DOT__fe1_onehot))));
    vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__ctrl_frac[0U] 
        = vlSelf->bf16_expe_equiv_top__DOT__u_poly4_dsp__DOT__s2_frac;
}

VL_INLINE_OPT void Vbf16_expe_equiv_top___024root___nba_comb__TOP__0(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___nba_comb__TOP__0\n"); );
    // Init
    CData/*7:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0;
    CData/*6:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0;
    SData/*8:0*/ bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp;
    bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0;
    // Body
    if ((3U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
         [0xdU])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (8U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (0x200000U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (0x800U | (0x3ff01fU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp));
    } else if ((1U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [0xdU])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (0x200fffU & vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (0x10U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
    } else if ((2U == vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__eo_delay
                [0xdU])) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = (1U | vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp);
    } else {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s8_final_fp 
            = vlSelf->bf16_expe_equiv_top__DOT__u_exp2_opt__DOT__s7_rounded_fp;
    }
    vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__base_count 
        = (0x1ffU & ((IData)(0x1f4U) + (((IData)(0x80U) 
                                         - (0x7fU & 
                                            ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_frac) 
                                             >> 9U))) 
                                        + (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s3_cand_dev))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[0U] 
        = (0xfffffffffULL & ((QData)((IData)((0x1fffffU 
                                              & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__packed_coeff)))) 
                             * (QData)((IData)((0x7fffU 
                                                & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__frac_aligned))))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[1U] 
        = (0xfffffffffULL & ((QData)((IData)((0x1fffffU 
                                              & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__packed_coeff)))) 
                             * (QData)((IData)((0x7fffU 
                                                & (IData)(
                                                          (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__frac_aligned 
                                                           >> 0xfU)))))));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__gen_split_mult__DOT__part_comb[2U] 
        = (0xfffffffffULL & ((QData)((IData)((0x1fffffU 
                                              & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__packed_coeff)))) 
                             * (QData)((IData)((0xffU 
                                                & (IData)(
                                                          (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_lin_approx__DOT__frac_aligned 
                                                           >> 0x1eU)))))));
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

VL_INLINE_OPT void Vbf16_expe_equiv_top___024root___nba_sequent__TOP__3(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___nba_sequent__TOP__3\n"); );
    // Body
    vlSelf->o_exp2opt_tready = (1U & ((~ (IData)(vlSelf->o_exp2opt_tvalid)) 
                                      | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_poly4dsp_tready = (1U & ((~ (IData)(vlSelf->o_poly4dsp_tvalid)) 
                                       | (IData)(vlSelf->m_axis_tready)));
    if (vlSelf->rst_n) {
        if (vlSelf->o_exp2_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_frac_part 
                = (0x3fffffffffULL & vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_unified_shift__DOT__unified_shifted);
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_unified_shift__DOT__unified_shifted 
                = (0x7fffffffffffULL & (VL_LTES_III(32, 0U, 
                                                    VL_EXTENDS_II(32,9, 
                                                                  vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                                                                  [3U]))
                                         ? VL_SHIFTL_QQI(47,47,9, 
                                                         ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_out)) 
                                                          << 9U), 
                                                         vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                                                         [3U])
                                         : VL_SHIFTR_QQI(47,47,9, 
                                                         ((QData)((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_out)) 
                                                          << 9U), 
                                                         (0x1ffU 
                                                          & (- 
                                                             vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay
                                                             [3U])))));
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_out 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__gen_rne_reg__DOT__mant_rne_q;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__gen_rne_reg__DOT__mant_rne_q 
                = ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__base2_d)
                    ? ((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__mant_src_d) 
                       << 0x16U) : vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__mant_mult);
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__mant_src_d 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_src;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__mant_mult 
                = (0x7fffffffU & ((IData)(0x5c551dU) 
                                  * (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_src)));
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__base2_d 
                = vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_base2;
            vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_base2 = 0U;
        }
        if (vlSelf->o_lut_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_addr 
                = ((0xf80U & (((vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed 
                                >> 0xcU) - (IData)(0x17U)) 
                              << 7U)) | (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed 
                                                  >> 5U)));
            vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))
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
                                               | (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))));
        }
        if (vlSelf->o_hybrid_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_dense_addr 
                = ((0x380U & (((IData)(1U) + (vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                                              >> 0xcU)) 
                              << 7U)) | (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                                                  >> 5U)));
            vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))
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
                                               | (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))));
        }
        if (vlSelf->o_cut_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_tail_addr 
                = vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_tail_addr;
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_tail_addr 
                = (0xfU & ((vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                            >> 5U) - (IData)(0xfU)));
            vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))
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
                                               | (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))));
        }
        if (vlSelf->o_poly4_tready) {
            vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb))
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
                                               | (((IData)(vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__u_decompose__DOT__status_comb)))));
        }
    } else {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s4_frac_part = 0ULL;
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s2_addr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s2_dense_addr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s2_tail_addr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_poly4__DOT__s1_decomposed = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_unified_shift__DOT__unified_shifted = 0ULL;
        vlSelf->bf16_expe_equiv_top__DOT__u_lut__DOT__s1_decomposed = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_hybrid__DOT__s1_decomposed = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__fe_tail_addr = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_out = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_cut__DOT__s1_decomposed = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__gen_rne_reg__DOT__mant_rne_q = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__mant_src_d = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__mant_mult = 0U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__u_log2e_mult__DOT__base2_d = 1U;
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_base2 = 1U;
    }
    vlSelf->o_poly4_tready = (1U & ((~ (IData)(vlSelf->o_poly4_tvalid)) 
                                    | (IData)(vlSelf->m_axis_tready)));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[0U] 
        = (0x1ffU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                     >> 0xcU));
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v0) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v1) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[1U] = 0U;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v2) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v3) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[2U] = 0U;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v4) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay__v5) {
        vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__exp_delay[3U] = 0U;
    }
    vlSelf->o_lut_tready = (1U & ((~ (IData)(vlSelf->o_lut_tvalid)) 
                                  | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_hybrid_tready = (1U & ((~ (IData)(vlSelf->o_hybrid_tvalid)) 
                                     | (IData)(vlSelf->m_axis_tready)));
    vlSelf->o_cut_tready = (1U & ((~ (IData)(vlSelf->o_cut_tvalid)) 
                                  | (IData)(vlSelf->m_axis_tready)));
    vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s3_mant_src 
        = ((0x80U & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                     << 3U)) | (0x7fU & (vlSelf->bf16_expe_equiv_top__DOT__u_exp2__DOT__s1_decomposed 
                                         >> 5U)));
    vlSelf->o_exp2_tready = (1U & ((~ (IData)(vlSelf->o_exp2_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
    vlSelf->s_axis_tready = ((IData)(vlSelf->o_exp2_tready) 
                             & ((IData)(vlSelf->o_exp2opt_tready) 
                                & ((IData)(vlSelf->o_lut_tready) 
                                   & ((IData)(vlSelf->o_hybrid_tready) 
                                      & ((IData)(vlSelf->o_cut_tready) 
                                         & ((IData)(vlSelf->o_poly4_tready) 
                                            & (IData)(vlSelf->o_poly4dsp_tready)))))));
}

void Vbf16_expe_equiv_top___024root___eval_nba(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_nba\n"); );
    // Body
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_equiv_top___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_equiv_top___024root___nba_sequent__TOP__1(vlSelf);
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_equiv_top___024root___nba_sequent__TOP__2(vlSelf);
    }
    if ((3ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_equiv_top___024root___nba_comb__TOP__0(vlSelf);
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_equiv_top___024root___nba_sequent__TOP__3(vlSelf);
    }
}

void Vbf16_expe_equiv_top___024root___eval_triggers__act(Vbf16_expe_equiv_top___024root* vlSelf);

bool Vbf16_expe_equiv_top___024root___eval_phase__act(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<2> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vbf16_expe_equiv_top___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vbf16_expe_equiv_top___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vbf16_expe_equiv_top___024root___eval_phase__nba(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vbf16_expe_equiv_top___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___dump_triggers__ico(Vbf16_expe_equiv_top___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___dump_triggers__nba(Vbf16_expe_equiv_top___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_equiv_top___024root___dump_triggers__act(Vbf16_expe_equiv_top___024root* vlSelf);
#endif  // VL_DEBUG

void Vbf16_expe_equiv_top___024root___eval(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval\n"); );
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
            Vbf16_expe_equiv_top___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("tests/bf16_expe_equiv_top.sv", 26, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vbf16_expe_equiv_top___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vbf16_expe_equiv_top___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("tests/bf16_expe_equiv_top.sv", 26, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vbf16_expe_equiv_top___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("tests/bf16_expe_equiv_top.sv", 26, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vbf16_expe_equiv_top___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vbf16_expe_equiv_top___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vbf16_expe_equiv_top___024root___eval_debug_assertions(Vbf16_expe_equiv_top___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_equiv_top__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_equiv_top___024root___eval_debug_assertions\n"); );
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
