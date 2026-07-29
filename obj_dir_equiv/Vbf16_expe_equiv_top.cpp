// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vbf16_expe_equiv_top__pch.h"

//============================================================
// Constructors

Vbf16_expe_equiv_top::Vbf16_expe_equiv_top(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vbf16_expe_equiv_top__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , s_axis_tvalid{vlSymsp->TOP.s_axis_tvalid}
    , s_axis_tready{vlSymsp->TOP.s_axis_tready}
    , m_axis_tready{vlSymsp->TOP.m_axis_tready}
    , o_exp2_tvalid{vlSymsp->TOP.o_exp2_tvalid}
    , o_exp2_tready{vlSymsp->TOP.o_exp2_tready}
    , o_exp2opt_tvalid{vlSymsp->TOP.o_exp2opt_tvalid}
    , o_exp2opt_tready{vlSymsp->TOP.o_exp2opt_tready}
    , o_lut_tvalid{vlSymsp->TOP.o_lut_tvalid}
    , o_lut_tready{vlSymsp->TOP.o_lut_tready}
    , o_hybrid_tvalid{vlSymsp->TOP.o_hybrid_tvalid}
    , o_hybrid_tready{vlSymsp->TOP.o_hybrid_tready}
    , o_cut_tvalid{vlSymsp->TOP.o_cut_tvalid}
    , o_cut_tready{vlSymsp->TOP.o_cut_tready}
    , o_poly4_tvalid{vlSymsp->TOP.o_poly4_tvalid}
    , o_poly4_tready{vlSymsp->TOP.o_poly4_tready}
    , o_poly4dsp_tvalid{vlSymsp->TOP.o_poly4dsp_tvalid}
    , o_poly4dsp_tready{vlSymsp->TOP.o_poly4dsp_tready}
    , s_axis_tdata{vlSymsp->TOP.s_axis_tdata}
    , o_exp2_tdata{vlSymsp->TOP.o_exp2_tdata}
    , o_exp2opt_tdata{vlSymsp->TOP.o_exp2opt_tdata}
    , o_lut_tdata{vlSymsp->TOP.o_lut_tdata}
    , o_hybrid_tdata{vlSymsp->TOP.o_hybrid_tdata}
    , o_cut_tdata{vlSymsp->TOP.o_cut_tdata}
    , o_poly4_tdata{vlSymsp->TOP.o_poly4_tdata}
    , o_poly4dsp_tdata{vlSymsp->TOP.o_poly4dsp_tdata}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vbf16_expe_equiv_top::Vbf16_expe_equiv_top(const char* _vcname__)
    : Vbf16_expe_equiv_top(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vbf16_expe_equiv_top::~Vbf16_expe_equiv_top() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vbf16_expe_equiv_top___024root___eval_debug_assertions(Vbf16_expe_equiv_top___024root* vlSelf);
#endif  // VL_DEBUG
void Vbf16_expe_equiv_top___024root___eval_static(Vbf16_expe_equiv_top___024root* vlSelf);
void Vbf16_expe_equiv_top___024root___eval_initial(Vbf16_expe_equiv_top___024root* vlSelf);
void Vbf16_expe_equiv_top___024root___eval_settle(Vbf16_expe_equiv_top___024root* vlSelf);
void Vbf16_expe_equiv_top___024root___eval(Vbf16_expe_equiv_top___024root* vlSelf);

void Vbf16_expe_equiv_top::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vbf16_expe_equiv_top::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vbf16_expe_equiv_top___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vbf16_expe_equiv_top___024root___eval_static(&(vlSymsp->TOP));
        Vbf16_expe_equiv_top___024root___eval_initial(&(vlSymsp->TOP));
        Vbf16_expe_equiv_top___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vbf16_expe_equiv_top___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vbf16_expe_equiv_top::eventsPending() { return false; }

uint64_t Vbf16_expe_equiv_top::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vbf16_expe_equiv_top::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vbf16_expe_equiv_top___024root___eval_final(Vbf16_expe_equiv_top___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_equiv_top::final() {
    Vbf16_expe_equiv_top___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vbf16_expe_equiv_top::hierName() const { return vlSymsp->name(); }
const char* Vbf16_expe_equiv_top::modelName() const { return "Vbf16_expe_equiv_top"; }
unsigned Vbf16_expe_equiv_top::threads() const { return 1; }
void Vbf16_expe_equiv_top::prepareClone() const { contextp()->prepareClone(); }
void Vbf16_expe_equiv_top::atClone() const {
    contextp()->threadPoolpOnClone();
}

//============================================================
// Trace configuration

VL_ATTR_COLD void Vbf16_expe_equiv_top::trace(VerilatedVcdC* tfp, int levels, int options) {
    vl_fatal(__FILE__, __LINE__, __FILE__,"'Vbf16_expe_equiv_top::trace()' called on model that was Verilated without --trace option");
}
