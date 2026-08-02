// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vbf16_expe_poly4.h for the primary calling header

#ifndef VERILATED_VBF16_EXPE_POLY4___024ROOT_H_
#define VERILATED_VBF16_EXPE_POLY4___024ROOT_H_  // guard

#include "verilated.h"


class Vbf16_expe_poly4__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vbf16_expe_poly4___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        VL_IN8(clk,0,0);
        VL_IN8(rst_n,0,0);
        VL_IN8(s_axis_tvalid,0,0);
        VL_OUT8(s_axis_tready,0,0);
        VL_OUT8(m_axis_tvalid,0,0);
        VL_IN8(m_axis_tready,0,0);
        CData/*1:0*/ bf16_expe_poly4__DOT__s2_route;
        CData/*1:0*/ bf16_expe_poly4__DOT__s2_eo;
        CData/*3:0*/ bf16_expe_poly4__DOT__s2_tail_addr;
        CData/*1:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_route;
        CData/*3:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_tail;
        CData/*1:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_eo;
        CData/*1:0*/ bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_route;
        CData/*3:0*/ bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_tail;
        CData/*1:0*/ bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_eo;
        CData/*3:0*/ bf16_expe_poly4__DOT__u_decompose__DOT__status_comb;
        CData/*0:0*/ bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v0;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v0;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v0;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v0;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v0;
        CData/*6:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v0;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v0;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v0;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v1;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v1;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v1;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v1;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v1;
        CData/*6:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v1;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v1;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v1;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v2;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v2;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v2;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v2;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v2;
        CData/*6:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v2;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v2;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v2;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v3;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v3;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v3;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v3;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v3;
        CData/*6:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v3;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v3;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v3;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v4;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v4;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v4;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v4;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v4;
        CData/*6:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v4;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v4;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v4;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v5;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v5;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v5;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v5;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v5;
        CData/*6:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v5;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v5;
    };
    struct {
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v5;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v6;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v6;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v6;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v6;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v6;
        CData/*6:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v6;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v6;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v6;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_eo__v7;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_eo__v7;
        CData/*1:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_route__v7;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_route__v7;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_int__v7;
        CData/*6:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_tail__v7;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_tail__v7;
        CData/*0:0*/ __Vdlyvset__bf16_expe_poly4__DOT__ctrl_frac__v7;
        CData/*0:0*/ __VstlFirstIteration;
        CData/*0:0*/ __VicoFirstIteration;
        CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
        CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
        CData/*0:0*/ __VactContinue;
        VL_IN16(s_axis_tdata,15,0);
        VL_OUT16(m_axis_tdata,15,0);
        SData/*8:0*/ bf16_expe_poly4__DOT__s2_int_part;
        SData/*15:0*/ bf16_expe_poly4__DOT__core_data;
        SData/*15:0*/ bf16_expe_poly4__DOT__gen_axi_pipelined__DOT__valid_sr;
        SData/*15:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_onehot;
        SData/*15:0*/ bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__0__KET____DOT__gen_head__DOT__head;
        SData/*15:0*/ bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
        SData/*8:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v0;
        SData/*8:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v1;
        SData/*8:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v2;
        SData/*8:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v3;
        SData/*8:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v4;
        SData/*8:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v5;
        SData/*8:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v6;
        SData/*8:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_int__v7;
        IData/*21:0*/ bf16_expe_poly4__DOT__s1_decomposed;
        IData/*16:0*/ bf16_expe_poly4__DOT__s2_frac;
        IData/*24:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_t7;
        IData/*25:0*/ bf16_expe_poly4__DOT__gen_fe2_reg__DOT__fe2_aligned;
        IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out;
        IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out;
        IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out;
        IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out;
        IData/*16:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v0;
        IData/*16:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v1;
        IData/*16:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v2;
        IData/*16:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v3;
        IData/*16:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v4;
        IData/*16:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v5;
        IData/*16:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v6;
        IData/*16:0*/ __Vdlyvval__bf16_expe_poly4__DOT__ctrl_frac__v7;
        IData/*31:0*/ __VactIterCount;
        QData/*42:0*/ bf16_expe_poly4__DOT__gen_horner__BRA__0__KET____DOT__u_step__DOT__mult_r;
        QData/*42:0*/ bf16_expe_poly4__DOT__gen_horner__BRA__1__KET____DOT__u_step__DOT__mult_r;
        QData/*42:0*/ bf16_expe_poly4__DOT__gen_horner__BRA__2__KET____DOT__u_step__DOT__mult_r;
        QData/*42:0*/ bf16_expe_poly4__DOT__gen_horner__BRA__3__KET____DOT__u_step__DOT__mult_r;
        VlUnpacked<CData/*1:0*/, 9> bf16_expe_poly4__DOT__ctrl_eo;
        VlUnpacked<CData/*1:0*/, 9> bf16_expe_poly4__DOT__ctrl_route;
        VlUnpacked<SData/*8:0*/, 9> bf16_expe_poly4__DOT__ctrl_int;
        VlUnpacked<CData/*6:0*/, 9> bf16_expe_poly4__DOT__ctrl_tail;
        VlUnpacked<IData/*16:0*/, 9> bf16_expe_poly4__DOT__ctrl_frac;
    };
    struct {
        VlUnpacked<CData/*6:0*/, 11> bf16_expe_poly4__DOT__u_tail_rom__DOT__rom;
        VlUnpacked<SData/*15:0*/, 3> bf16_expe_poly4__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain;
    };
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vbf16_expe_poly4__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vbf16_expe_poly4___024root(Vbf16_expe_poly4__Syms* symsp, const char* v__name);
    ~Vbf16_expe_poly4___024root();
    VL_UNCOPYABLE(Vbf16_expe_poly4___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
