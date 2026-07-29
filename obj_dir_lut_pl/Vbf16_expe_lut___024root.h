// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vbf16_expe_lut.h for the primary calling header

#ifndef VERILATED_VBF16_EXPE_LUT___024ROOT_H_
#define VERILATED_VBF16_EXPE_LUT___024ROOT_H_  // guard

#include "verilated.h"


class Vbf16_expe_lut__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vbf16_expe_lut___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(s_axis_tvalid,0,0);
    VL_OUT8(s_axis_tready,0,0);
    VL_OUT8(m_axis_tvalid,0,0);
    VL_IN8(m_axis_tready,0,0);
    CData/*1:0*/ bf16_expe_lut__DOT__s2_eo_code;
    CData/*3:0*/ bf16_expe_lut__DOT__u_decompose__DOT__status_comb;
    CData/*0:0*/ bf16_expe_lut__DOT__u_decompose__DOT__hidden_bit_comb;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    VL_IN16(s_axis_tdata,15,0);
    VL_OUT16(m_axis_tdata,15,0);
    SData/*11:0*/ bf16_expe_lut__DOT__s2_addr;
    SData/*15:0*/ bf16_expe_lut__DOT__s3_rom_data;
    SData/*15:0*/ bf16_expe_lut__DOT__core_data;
    SData/*15:0*/ bf16_expe_lut__DOT__gen_axi_pipelined__DOT__vld_sr;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__1__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__2__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__3__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__4__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__5__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__6__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__7__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__8__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__9__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__10__KET____DOT__nxt;
    SData/*15:0*/ bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__gen_stage__BRA__11__KET____DOT__nxt;
    IData/*21:0*/ bf16_expe_lut__DOT__s1_decomposed;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<CData/*1:0*/, 2> bf16_expe_lut__DOT__eo_delay;
    VlUnpacked<SData/*15:0*/, 2176> bf16_expe_lut__DOT__u_rom__DOT__rom;
    VlUnpacked<SData/*15:0*/, 12> bf16_expe_lut__DOT__u_pipe_pad__DOT__gen_delay__DOT__chain;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vbf16_expe_lut__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vbf16_expe_lut___024root(Vbf16_expe_lut__Syms* symsp, const char* v__name);
    ~Vbf16_expe_lut___024root();
    VL_UNCOPYABLE(Vbf16_expe_lut___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
