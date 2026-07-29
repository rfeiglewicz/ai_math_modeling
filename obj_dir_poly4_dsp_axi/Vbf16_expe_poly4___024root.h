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
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(s_axis_tvalid,0,0);
    VL_OUT8(s_axis_tready,0,0);
    VL_OUT8(m_axis_tvalid,0,0);
    VL_IN8(m_axis_tready,0,0);
    CData/*1:0*/ bf16_expe_poly4__DOT__s2_route;
    CData/*1:0*/ bf16_expe_poly4__DOT__s2_eo;
    CData/*3:0*/ bf16_expe_poly4__DOT__s2_tail_addr;
    CData/*7:0*/ bf16_expe_poly4__DOT__gen_axi_pipelined__DOT__valid_sr;
    CData/*1:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_route;
    CData/*3:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_tail;
    CData/*1:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_eo;
    CData/*3:0*/ bf16_expe_poly4__DOT__u_decompose__DOT__status_comb;
    CData/*0:0*/ bf16_expe_poly4__DOT__u_decompose__DOT__hidden_bit_comb;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    VL_IN16(s_axis_tdata,15,0);
    VL_OUT16(m_axis_tdata,15,0);
    SData/*8:0*/ bf16_expe_poly4__DOT__s2_int_part;
    SData/*15:0*/ bf16_expe_poly4__DOT__core_data;
    SData/*15:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_onehot;
    IData/*21:0*/ bf16_expe_poly4__DOT__s1_decomposed;
    IData/*16:0*/ bf16_expe_poly4__DOT__s2_frac;
    IData/*24:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__fe1_t7;
    IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__0__KET____DOT__u_step__acc_out;
    IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__1__KET____DOT__u_step__acc_out;
    IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__2__KET____DOT__u_step__acc_out;
    IData/*24:0*/ bf16_expe_poly4__DOT____Vcellout__gen_horner__BRA__3__KET____DOT__u_step__acc_out;
    IData/*31:0*/ __VactIterCount;
    QData/*40:0*/ bf16_expe_poly4__DOT__gen_frontend_dsp__DOT__scaled_comb;
    VlUnpacked<CData/*1:0*/, 5> bf16_expe_poly4__DOT__ctrl_eo;
    VlUnpacked<CData/*1:0*/, 5> bf16_expe_poly4__DOT__ctrl_route;
    VlUnpacked<SData/*8:0*/, 5> bf16_expe_poly4__DOT__ctrl_int;
    VlUnpacked<CData/*6:0*/, 5> bf16_expe_poly4__DOT__ctrl_tail;
    VlUnpacked<IData/*16:0*/, 5> bf16_expe_poly4__DOT__ctrl_frac;
    VlUnpacked<CData/*6:0*/, 11> bf16_expe_poly4__DOT__u_tail_rom__DOT__rom;
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
