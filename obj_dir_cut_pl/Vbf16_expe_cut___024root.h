// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vbf16_expe_cut.h for the primary calling header

#ifndef VERILATED_VBF16_EXPE_CUT___024ROOT_H_
#define VERILATED_VBF16_EXPE_CUT___024ROOT_H_  // guard

#include "verilated.h"


class Vbf16_expe_cut__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vbf16_expe_cut___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(s_axis_tvalid,0,0);
    VL_OUT8(s_axis_tready,0,0);
    VL_OUT8(m_axis_tvalid,0,0);
    VL_IN8(m_axis_tready,0,0);
    CData/*1:0*/ bf16_expe_cut__DOT__s2_eo_code;
    CData/*4:0*/ bf16_expe_cut__DOT__fe_shift;
    CData/*1:0*/ bf16_expe_cut__DOT__fe_route;
    CData/*3:0*/ bf16_expe_cut__DOT__fe_tail_addr;
    CData/*1:0*/ bf16_expe_cut__DOT__fe_eo_code;
    CData/*1:0*/ bf16_expe_cut__DOT__s2_route;
    CData/*3:0*/ bf16_expe_cut__DOT__s2_tail_addr;
    CData/*3:0*/ bf16_expe_cut__DOT__s3_cand_dev;
    CData/*6:0*/ bf16_expe_cut__DOT__s3_tail_data;
    CData/*1:0*/ bf16_expe_cut__DOT__s3_eo_code;
    CData/*1:0*/ bf16_expe_cut__DOT__s3_route;
    CData/*0:0*/ bf16_expe_cut__DOT__s4_cut_sentinel;
    CData/*1:0*/ bf16_expe_cut__DOT__s4_eo_code;
    CData/*1:0*/ bf16_expe_cut__DOT__s4_route;
    CData/*6:0*/ bf16_expe_cut__DOT__s4_tail_data;
    CData/*0:0*/ bf16_expe_cut__DOT__s5_below_cut;
    CData/*1:0*/ bf16_expe_cut__DOT__s5_eo_code;
    CData/*1:0*/ bf16_expe_cut__DOT__s5_route;
    CData/*6:0*/ bf16_expe_cut__DOT__s5_tail_data;
    CData/*3:0*/ bf16_expe_cut__DOT__u_decompose__DOT__status_comb;
    CData/*0:0*/ bf16_expe_cut__DOT__u_decompose__DOT__hidden_bit_comb;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    VL_IN16(s_axis_tdata,15,0);
    VL_OUT16(m_axis_tdata,15,0);
    SData/*8:0*/ bf16_expe_cut__DOT__s2_int_part;
    SData/*15:0*/ bf16_expe_cut__DOT__s2_frac;
    SData/*8:0*/ bf16_expe_cut__DOT__s3_int_part;
    SData/*15:0*/ bf16_expe_cut__DOT__s3_frac;
    SData/*8:0*/ bf16_expe_cut__DOT__base_count;
    SData/*8:0*/ bf16_expe_cut__DOT__s4_base_count;
    SData/*12:0*/ bf16_expe_cut__DOT__s4_cut_dev;
    SData/*15:0*/ bf16_expe_cut__DOT__s4_frac;
    SData/*8:0*/ bf16_expe_cut__DOT__s4_int_part;
    SData/*8:0*/ bf16_expe_cut__DOT__ladder_count;
    SData/*8:0*/ bf16_expe_cut__DOT__s5_base_count;
    SData/*8:0*/ bf16_expe_cut__DOT__s5_int_part;
    SData/*15:0*/ bf16_expe_cut__DOT__core_data;
    SData/*11:0*/ bf16_expe_cut__DOT__gen_axi_pipelined__DOT__valid_sr;
    SData/*15:0*/ bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt;
    IData/*21:0*/ bf16_expe_cut__DOT__s1_decomposed;
    IData/*31:0*/ bf16_expe_cut__DOT__fe_prod;
    IData/*31:0*/ bf16_expe_cut__DOT____VdfgTmp_h0f01a197__0;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<CData/*3:0*/, 256> bf16_expe_cut__DOT__u_cand_rom__DOT__rom;
    VlUnpacked<CData/*6:0*/, 11> bf16_expe_cut__DOT__u_tail_rom__DOT__rom;
    VlUnpacked<SData/*12:0*/, 128> bf16_expe_cut__DOT__u_cut_rom__DOT__rom;
    VlUnpacked<SData/*15:0*/, 5> bf16_expe_cut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vbf16_expe_cut__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vbf16_expe_cut___024root(Vbf16_expe_cut__Syms* symsp, const char* v__name);
    ~Vbf16_expe_cut___024root();
    VL_UNCOPYABLE(Vbf16_expe_cut___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
