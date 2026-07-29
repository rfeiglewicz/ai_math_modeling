// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vbf16_expe_hybrid.h for the primary calling header

#ifndef VERILATED_VBF16_EXPE_HYBRID___024ROOT_H_
#define VERILATED_VBF16_EXPE_HYBRID___024ROOT_H_  // guard

#include "verilated.h"


class Vbf16_expe_hybrid__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vbf16_expe_hybrid___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(s_axis_tvalid,0,0);
    VL_OUT8(s_axis_tready,0,0);
    VL_OUT8(m_axis_tvalid,0,0);
    VL_IN8(m_axis_tready,0,0);
    CData/*1:0*/ bf16_expe_hybrid__DOT__s2_eo_code;
    CData/*1:0*/ bf16_expe_hybrid__DOT__s2_route;
    CData/*2:0*/ bf16_expe_hybrid__DOT__s2_sparse_exp_index;
    CData/*6:0*/ bf16_expe_hybrid__DOT__s2_mantissa;
    CData/*1:0*/ bf16_expe_hybrid__DOT__s3_eo_code;
    CData/*1:0*/ bf16_expe_hybrid__DOT__s3_route;
    CData/*3:0*/ bf16_expe_hybrid__DOT__gen_axi_pipelined__DOT__valid_sr;
    CData/*3:0*/ bf16_expe_hybrid__DOT__u_decompose__DOT__status_comb;
    CData/*0:0*/ bf16_expe_hybrid__DOT__u_decompose__DOT__hidden_bit_comb;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    VL_IN16(s_axis_tdata,15,0);
    VL_OUT16(m_axis_tdata,15,0);
    SData/*9:0*/ bf16_expe_hybrid__DOT__s2_dense_addr;
    SData/*15:0*/ bf16_expe_hybrid__DOT__s3_sparse_data;
    SData/*15:0*/ bf16_expe_hybrid__DOT__s3_dense_data;
    SData/*15:0*/ bf16_expe_hybrid__DOT__u_sparse__DOT__data_comb;
    IData/*21:0*/ bf16_expe_hybrid__DOT__s1_decomposed;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<SData/*15:0*/, 1024> bf16_expe_hybrid__DOT__u_dense_rom__DOT__rom;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vbf16_expe_hybrid__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vbf16_expe_hybrid___024root(Vbf16_expe_hybrid__Syms* symsp, const char* v__name);
    ~Vbf16_expe_hybrid___024root();
    VL_UNCOPYABLE(Vbf16_expe_hybrid___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
