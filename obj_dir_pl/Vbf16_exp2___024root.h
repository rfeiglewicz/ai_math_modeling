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
    CData/*1:0*/ bf16_exp2__DOT__s2_eo_code;
    CData/*0:0*/ bf16_exp2__DOT__s3_base2;
    CData/*7:0*/ bf16_exp2__DOT__s3_mant_src;
    CData/*6:0*/ bf16_exp2__DOT__gen_axi_pipelined__DOT__vld_sr;
    CData/*3:0*/ bf16_exp2__DOT__u_decompose__DOT__status_comb;
    CData/*0:0*/ bf16_exp2__DOT__u_decompose__DOT__hidden_bit_comb;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    VL_IN16(s_axis_tdata,15,0);
    VL_OUT16(m_axis_tdata,15,0);
    SData/*8:0*/ bf16_exp2__DOT__s4_exponent;
    SData/*8:0*/ bf16_exp2__DOT__s5_int_part;
    SData/*8:0*/ bf16_exp2__DOT__s6_int_part;
    SData/*8:0*/ bf16_exp2__DOT__s4_int_part;
    SData/*8:0*/ bf16_exp2__DOT__s6_poly_exp;
    SData/*15:0*/ bf16_exp2__DOT__bf16_out;
    SData/*8:0*/ bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb;
    SData/*15:0*/ bf16_exp2__DOT__u_recompose__DOT__bf16_comb;
    IData/*21:0*/ bf16_exp2__DOT__s1_decomposed;
    IData/*30:0*/ bf16_exp2__DOT__s3_mant_out;
    IData/*21:0*/ bf16_exp2__DOT__s7_rounded_fp;
    IData/*21:0*/ bf16_exp2__DOT__s8_final_fp;
    IData/*21:0*/ bf16_exp2__DOT__u_round__DOT__rounded_comb;
    IData/*31:0*/ __VactIterCount;
    QData/*37:0*/ bf16_exp2__DOT__s4_frac_part;
    QData/*58:0*/ bf16_exp2__DOT__s6_norm_mant;
    QData/*46:0*/ bf16_exp2__DOT__u_unified_shift__DOT__unified_shifted;
    QData/*41:0*/ bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff;
    QData/*37:0*/ bf16_exp2__DOT__u_lin_approx__DOT__frac_aligned;
    QData/*58:0*/ bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb;
    QData/*58:0*/ bf16_exp2__DOT__u_round__DOT__unnamedblk1__DOT__sticky_mask;
    VlUnpacked<CData/*1:0*/, 5> bf16_exp2__DOT__eo_delay;
    VlUnpacked<QData/*41:0*/, 128> bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

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
