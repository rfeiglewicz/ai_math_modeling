// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_poly4.h for the primary calling header

#include "Vbf16_expe_poly4__pch.h"
#include "Vbf16_expe_poly4___024root.h"

VL_INLINE_OPT void Vbf16_expe_poly4___024root___ico_sequent__TOP__0(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
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
    vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
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

VL_INLINE_OPT void Vbf16_expe_poly4___024root___nba_sequent__TOP__0(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___nba_sequent__TOP__0\n"); );
    // Body
    if (vlSelf->rst_n) {
        if (vlSelf->s_axis_tready) {
            vlSelf->bf16_expe_poly4__DOT__gen_axi_pipelined__DOT__valid_sr 
                = ((0xfffeU & ((IData)(vlSelf->bf16_expe_poly4__DOT__gen_axi_pipelined__DOT__valid_sr) 
                               << 1U)) | (IData)(vlSelf->s_axis_tvalid));
        }
    } else {
        vlSelf->bf16_expe_poly4__DOT__gen_axi_pipelined__DOT__valid_sr = 0U;
    }
    vlSelf->m_axis_tvalid = (1U & ((IData)(vlSelf->bf16_expe_poly4__DOT__gen_axi_pipelined__DOT__valid_sr) 
                                   >> 0xfU));
}

VL_INLINE_OPT void Vbf16_expe_poly4___024root___nba_sequent__TOP__1(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___nba_sequent__TOP__1\n"); );
    // Init
    SData/*15:0*/ __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0;
    SData/*15:0*/ __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0;
    // Body
    __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 0U;
    __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v7 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v7 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v7 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v7 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v0 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v1 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v2 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v3 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v4 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v5 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v6 = 0U;
    vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v7 = 0U;
    __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 0U;
    if (vlSelf->s_axis_tready) {
        __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 
            = vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1 = 1U;
        __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 
            = vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
        __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v0 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_int
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v1 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_int
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v2 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_int
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v3 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_int
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v4 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_int
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v5 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_int
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v6 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_int
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v7 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_int
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v7 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v0 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_route
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v1 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_route
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v2 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_route
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v3 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_route
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v4 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_route
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v5 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_route
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v6 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_route
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v7 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_route
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v7 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v0 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_eo
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v1 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_eo
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v2 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_eo
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v3 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_eo
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v4 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_eo
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v5 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_eo
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v6 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_eo
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v7 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_eo
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v7 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v0 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_tail
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v1 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_tail
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v2 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_tail
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v3 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_tail
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v4 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_tail
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v5 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_tail
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v6 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_tail
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v7 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_tail
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v7 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v0 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_frac
            [0U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v0 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v1 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_frac
            [1U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v1 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v2 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_frac
            [2U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v2 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v3 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_frac
            [3U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v3 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v4 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_frac
            [4U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v4 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v5 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_frac
            [5U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v5 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v6 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_frac
            [6U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v6 = 1U;
        vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v7 
            = vlSelf->bf16_expe_poly4__DOT__ctrl_frac
            [7U];
        vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v7 = 1U;
        __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 
            = vlSelf->bf16_expe_poly4__DOT__core_data;
        __Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0 = 1U;
        vlSelf->bf16_expe_poly4__DOT__s2_eo = vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_eo;
        vlSelf->bf16_expe_poly4__DOT__s2_route = vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_route;
        vlSelf->bf16_expe_poly4__DOT__s2_tail_addr 
            = vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_tail;
        vlSelf->bf16_expe_poly4__DOT__s2_int_part = 
            (0x1ffU & (vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_aligned 
                       >> 0x11U));
        vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_eo 
            = vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_eo;
        vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_route 
            = vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_route;
        vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_tail 
            = vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_tail;
        vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_eo 
            = ((8U & vlSelf->bf16_expe_poly4__DOT__s1_decomposed)
                ? 3U : ((1U & vlSelf->bf16_expe_poly4__DOT__s1_decomposed)
                         ? 1U : ((4U & vlSelf->bf16_expe_poly4__DOT__s1_decomposed)
                                  ? ((0x200000U & vlSelf->bf16_expe_poly4__DOT__s1_decomposed)
                                      ? 2U : 1U) : 
                                 ((0x200000U & vlSelf->bf16_expe_poly4__DOT__s1_decomposed)
                                   ? (VL_GTS_III(9, 0x1f7U, 
                                                 (0x1ffU 
                                                  & (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                                     >> 0xcU)))
                                       ? 1U : (VL_LTS_III(9, 7U, 
                                                          (0x1ffU 
                                                           & (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                                              >> 0xcU)))
                                                ? 2U
                                                : 0U))
                                   : 1U))));
        vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_route 
            = (VL_LTS_III(9, 6U, (0x1ffU & (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                            >> 0xcU)))
                ? 2U : ((IData)(((0x6000U == (0x1ff000U 
                                              & vlSelf->bf16_expe_poly4__DOT__s1_decomposed)) 
                                 & (0x3aU <= (0x7fU 
                                              & (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                                 >> 5U)))))
                         ? 2U : ((IData)(((0x6000U 
                                           == (0x1ff000U 
                                               & vlSelf->bf16_expe_poly4__DOT__s1_decomposed)) 
                                          & (0x2fU 
                                             <= (0x7fU 
                                                 & (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                                    >> 5U)))))
                                  ? 1U : 0U)));
        vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_tail 
            = (0xfU & ((vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                        >> 5U) - (IData)(0xfU)));
    }
    if (__Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0) {
        vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[0U] 
            = __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v0;
    }
    if (__Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1) {
        vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[1U] 
            = __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v1;
    }
    if (__Vdlyvset__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2) {
        vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain[2U] 
            = __Vdlyvval__bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain__v2;
    }
    vlSelf->m_axis_tdata = vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [2U];
    vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt 
        = vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [0U];
    vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt 
        = vlSelf->bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain
        [1U];
}

VL_INLINE_OPT void Vbf16_expe_poly4___024root___nba_sequent__TOP__2(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___nba_sequent__TOP__2\n"); );
    // Body
    if (vlSelf->rst_n) {
        if (vlSelf->s_axis_tready) {
            vlSelf->bf16_expe_poly4__DOT__core_data 
                = ((3U == vlSelf->bf16_expe_poly4__DOT__ctrl_eo
                    [8U]) ? 0xffc0U : ((1U == vlSelf->bf16_expe_poly4__DOT__ctrl_eo
                                        [8U]) ? 0x3f80U
                                        : ((2U == vlSelf->bf16_expe_poly4__DOT__ctrl_eo
                                            [8U]) ? 0U
                                            : ((0U 
                                                == 
                                                vlSelf->bf16_expe_poly4__DOT__ctrl_route
                                                [8U])
                                                ? (
                                                   (0x7f80U 
                                                    & ((((IData)(0x7eU) 
                                                         - 
                                                         vlSelf->bf16_expe_poly4__DOT__ctrl_int
                                                         [8U]) 
                                                        + 
                                                        VL_SHIFTR_III(8,8,32, 
                                                                      (0xffU 
                                                                       & VL_SHIFTRS_III(25,25,32, vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out, 0xfU)), 7U)) 
                                                       << 7U)) 
                                                   | (0x7fU 
                                                      & VL_SHIFTRS_III(25,25,32, vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out, 0xfU)))
                                                : (
                                                   (1U 
                                                    == 
                                                    vlSelf->bf16_expe_poly4__DOT__ctrl_route
                                                    [8U])
                                                    ? 
                                                   vlSelf->bf16_expe_poly4__DOT__ctrl_tail
                                                   [8U]
                                                    : 0U)))));
        }
    } else {
        vlSelf->bf16_expe_poly4__DOT__core_data = 0U;
    }
}

VL_INLINE_OPT void Vbf16_expe_poly4___024root___nba_sequent__TOP__3(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___nba_sequent__TOP__3\n"); );
    // Body
    vlSelf->bf16_expe_poly4__DOT__ctrl_eo[0U] = vlSelf->bf16_expe_poly4__DOT__s2_eo;
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[1U] = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v1) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[2U] = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v2) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[3U] = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v3) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[4U] = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v4) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[5U] = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v5) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[6U] = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v6) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[7U] = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v7) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_eo[8U] = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v7;
    }
    vlSelf->bf16_expe_poly4__DOT__ctrl_route[0U] = vlSelf->bf16_expe_poly4__DOT__s2_route;
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v1) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v2) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v3) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v4) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v5) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v6) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v7) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_route[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v7;
    }
    vlSelf->bf16_expe_poly4__DOT__ctrl_int[0U] = vlSelf->bf16_expe_poly4__DOT__s2_int_part;
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v1) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v2) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v3) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v4) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v5) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v6) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v7) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_int[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v7;
    }
    vlSelf->bf16_expe_poly4__DOT__ctrl_tail[0U] = (
                                                   (0xaU 
                                                    >= (IData)(vlSelf->bf16_expe_poly4__DOT__s2_tail_addr))
                                                    ? 
                                                   vlSelf->bf16_expe_poly4__DOT__u_tail_rom__DOT__rom
                                                   [vlSelf->bf16_expe_poly4__DOT__s2_tail_addr]
                                                    : 0U);
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v1) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v2) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v3) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v4) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v5) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v6) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v7) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_tail[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v7;
    }
    if (vlSelf->s_axis_tready) {
        vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x68080010000ULL 
                                                          + vlSelf->bf16_expe_poly4__DOT__gen_horner__BRA__3__KET____DOT__u_step__DOT__mult_r)), 0x11U))));
        vlSelf->bf16_expe_poly4__DOT__gen_horner__BRA__3__KET____DOT__u_step__DOT__mult_r 
            = (0x7ffffffffffULL & VL_MULS_QQQ(43, (0x7ffffffffffULL 
                                                   & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out)), 
                                              (0x7ffffffffffULL 
                                               & VL_EXTENDS_QI(43,18, 
                                                               vlSelf->bf16_expe_poly4__DOT__ctrl_frac
                                                               [6U]))));
        vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x74e92010000ULL 
                                                          + vlSelf->bf16_expe_poly4__DOT__gen_horner__BRA__2__KET____DOT__u_step__DOT__mult_r)), 0x11U))));
        vlSelf->bf16_expe_poly4__DOT__gen_horner__BRA__2__KET____DOT__u_step__DOT__mult_r 
            = (0x7ffffffffffULL & VL_MULS_QQQ(43, (0x7ffffffffffULL 
                                                   & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out)), 
                                              (0x7ffffffffffULL 
                                               & VL_EXTENDS_QI(43,18, 
                                                               vlSelf->bf16_expe_poly4__DOT__ctrl_frac
                                                               [4U]))));
        vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x3d54010000ULL 
                                                          + vlSelf->bf16_expe_poly4__DOT__gen_horner__BRA__1__KET____DOT__u_step__DOT__mult_r)), 0x11U))));
        vlSelf->bf16_expe_poly4__DOT__gen_horner__BRA__1__KET____DOT__u_step__DOT__mult_r 
            = (0x7ffffffffffULL & VL_MULS_QQQ(43, (0x7ffffffffffULL 
                                                   & VL_EXTENDS_QI(43,25, vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out)), 
                                              (0x7ffffffffffULL 
                                               & VL_EXTENDS_QI(43,18, 
                                                               vlSelf->bf16_expe_poly4__DOT__ctrl_frac
                                                               [2U]))));
        vlSelf->bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out 
            = (0x1ffffffU & (IData)((0x7ffffffffffULL 
                                     & VL_SHIFTRS_QQI(43,43,32, 
                                                      (0x7ffffffffffULL 
                                                       & (0x7f252010000ULL 
                                                          + vlSelf->bf16_expe_poly4__DOT__gen_horner__BRA__0__KET____DOT__u_step__DOT__mult_r)), 0x11U))));
        vlSelf->bf16_expe_poly4__DOT__gen_horner__BRA__0__KET____DOT__u_step__DOT__mult_r 
            = (0x7ffffffffffULL & VL_MULS_QQQ(43, 0xe400ULL, 
                                              (0x7ffffffffffULL 
                                               & VL_EXTENDS_QI(43,18, vlSelf->bf16_expe_poly4__DOT__s2_frac))));
        vlSelf->bf16_expe_poly4__DOT__s2_frac = (0x1ffffU 
                                                 & vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_aligned);
        vlSelf->bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_aligned 
            = (0x3ffffffU & (IData)((0x3ffffffULL & 
                                     (((QData)((IData)(vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_t7)) 
                                       * (QData)((IData)(vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_onehot))) 
                                      >> 0xfU))));
        vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_t7 
            = (((IData)(0xb8aa3bU) * (0x80U | (0x7fU 
                                               & (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                                  >> 5U)))) 
               >> 7U);
        vlSelf->bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_onehot 
            = (0xffffU & ((IData)(1U) << (0xfU & ((IData)(9U) 
                                                  + 
                                                  (vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                                                   >> 0xcU)))));
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v0) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[1U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v0;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v1) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[2U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v1;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v2) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[3U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v2;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v3) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[4U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v3;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v4) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[5U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v4;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v5) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[6U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v5;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v6) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[7U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v6;
    }
    if (vlSelf->__Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v7) {
        vlSelf->bf16_expe_poly4__DOT__ctrl_frac[8U] 
            = vlSelf->__Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v7;
    }
    vlSelf->bf16_expe_poly4__DOT__ctrl_frac[0U] = vlSelf->bf16_expe_poly4__DOT__s2_frac;
}

VL_INLINE_OPT void Vbf16_expe_poly4___024root___nba_sequent__TOP__4(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___nba_sequent__TOP__4\n"); );
    // Body
    if (vlSelf->rst_n) {
        if (vlSelf->s_axis_tready) {
            vlSelf->bf16_expe_poly4__DOT__s1_decomposed 
                = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                 << 6U)) | ((((IData)(
                                                      (0U 
                                                       != 
                                                       (0xdU 
                                                        & (IData)(vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb))))
                                               ? 0U
                                               : ((2U 
                                                   & (IData)(vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb))
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
                                               | (((IData)(vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb) 
                                                   << 4U) 
                                                  | (IData)(vlSelf->bf16_expe_poly4__DOT__u_decompose__DOT__status_comb)))));
        }
    } else {
        vlSelf->bf16_expe_poly4__DOT__s1_decomposed = 0U;
    }
    vlSelf->s_axis_tready = (1U & ((~ (IData)(vlSelf->m_axis_tvalid)) 
                                   | (IData)(vlSelf->m_axis_tready)));
}

void Vbf16_expe_poly4___024root___eval_nba(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_poly4___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_poly4___024root___nba_sequent__TOP__1(vlSelf);
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_poly4___024root___nba_sequent__TOP__2(vlSelf);
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_poly4___024root___nba_sequent__TOP__3(vlSelf);
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vbf16_expe_poly4___024root___nba_sequent__TOP__4(vlSelf);
    }
}

void Vbf16_expe_poly4___024root___eval_triggers__act(Vbf16_expe_poly4___024root* vlSelf);

bool Vbf16_expe_poly4___024root___eval_phase__act(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<2> __VpreTriggered;
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
            VL_FATAL_MT("src/rtl/bf16_expe_poly4.sv", 121, "", "Input combinational region did not converge.");
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
            VL_FATAL_MT("src/rtl/bf16_expe_poly4.sv", 121, "", "NBA region did not converge.");
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
                VL_FATAL_MT("src/rtl/bf16_expe_poly4.sv", 121, "", "Active region did not converge.");
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
