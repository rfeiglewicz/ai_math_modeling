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
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __VactContinue;
    VL_IN16(s_axis_tdata,15,0);
    VL_OUT16(m_axis_tdata,15,0);
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<CData/*1:0*/, 5> bf16_expe_poly4__DOT__ctrl_eo;
    VlUnpacked<CData/*1:0*/, 5> bf16_expe_poly4__DOT__ctrl_route;
    VlUnpacked<CData/*6:0*/, 5> bf16_expe_poly4__DOT__ctrl_tail;
    VlUnpacked<CData/*6:0*/, 11> bf16_expe_poly4__DOT__u_tail_rom__DOT__rom;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<0> __VactTriggered;
    VlTriggerVec<0> __VnbaTriggered;

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
