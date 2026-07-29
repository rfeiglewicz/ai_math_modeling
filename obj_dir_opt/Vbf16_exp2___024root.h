// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vbf16_exp2.h for the primary calling header

#ifndef VERILATED_VBF16_EXP2___024ROOT_H_
#define VERILATED_VBF16_EXP2___024ROOT_H_  // guard

#include "verilated.h"


class Vbf16_exp2__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vbf16_exp2___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(s_axis_tuser,0,0);
    VL_IN8(s_axis_tvalid,0,0);
    VL_OUT8(s_axis_tready,0,0);
    VL_OUT8(m_axis_tvalid,0,0);
    VL_IN8(m_axis_tready,0,0);
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __VactContinue;
    VL_IN16(s_axis_tdata,15,0);
    VL_OUT16(m_axis_tdata,15,0);
    IData/*21:0*/ bf16_exp2__DOT__s8_final_fp;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<QData/*41:0*/, 128> bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<0> __VactTriggered;
    VlTriggerVec<0> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vbf16_exp2__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vbf16_exp2___024root(Vbf16_exp2__Syms* symsp, const char* v__name);
    ~Vbf16_exp2___024root();
    VL_UNCOPYABLE(Vbf16_exp2___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
