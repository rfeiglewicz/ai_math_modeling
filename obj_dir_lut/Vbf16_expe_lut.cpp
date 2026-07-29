// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vbf16_expe_lut__pch.h"

//============================================================
// Constructors

Vbf16_expe_lut::Vbf16_expe_lut(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vbf16_expe_lut__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , s_axis_tvalid{vlSymsp->TOP.s_axis_tvalid}
    , s_axis_tready{vlSymsp->TOP.s_axis_tready}
    , m_axis_tvalid{vlSymsp->TOP.m_axis_tvalid}
    , m_axis_tready{vlSymsp->TOP.m_axis_tready}
    , s_axis_tdata{vlSymsp->TOP.s_axis_tdata}
    , m_axis_tdata{vlSymsp->TOP.m_axis_tdata}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vbf16_expe_lut::Vbf16_expe_lut(const char* _vcname__)
    : Vbf16_expe_lut(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vbf16_expe_lut::~Vbf16_expe_lut() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vbf16_expe_lut___024root___eval_debug_assertions(Vbf16_expe_lut___024root* vlSelf);
#endif  // VL_DEBUG
void Vbf16_expe_lut___024root___eval_static(Vbf16_expe_lut___024root* vlSelf);
void Vbf16_expe_lut___024root___eval_initial(Vbf16_expe_lut___024root* vlSelf);
void Vbf16_expe_lut___024root___eval_settle(Vbf16_expe_lut___024root* vlSelf);
void Vbf16_expe_lut___024root___eval(Vbf16_expe_lut___024root* vlSelf);

void Vbf16_expe_lut::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vbf16_expe_lut::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vbf16_expe_lut___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vbf16_expe_lut___024root___eval_static(&(vlSymsp->TOP));
        Vbf16_expe_lut___024root___eval_initial(&(vlSymsp->TOP));
        Vbf16_expe_lut___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vbf16_expe_lut___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vbf16_expe_lut::eventsPending() { return false; }

uint64_t Vbf16_expe_lut::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vbf16_expe_lut::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vbf16_expe_lut___024root___eval_final(Vbf16_expe_lut___024root* vlSelf);

VL_ATTR_COLD void Vbf16_expe_lut::final() {
    Vbf16_expe_lut___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vbf16_expe_lut::hierName() const { return vlSymsp->name(); }
const char* Vbf16_expe_lut::modelName() const { return "Vbf16_expe_lut"; }
unsigned Vbf16_expe_lut::threads() const { return 1; }
void Vbf16_expe_lut::prepareClone() const { contextp()->prepareClone(); }
void Vbf16_expe_lut::atClone() const {
    contextp()->threadPoolpOnClone();
}

//============================================================
// Trace configuration

VL_ATTR_COLD void Vbf16_expe_lut::trace(VerilatedVcdC* tfp, int levels, int options) {
    vl_fatal(__FILE__, __LINE__, __FILE__,"'Vbf16_expe_lut::trace()' called on model that was Verilated without --trace option");
}
