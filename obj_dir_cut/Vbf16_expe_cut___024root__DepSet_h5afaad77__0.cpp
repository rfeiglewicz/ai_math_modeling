// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_cut.h for the primary calling header

#include "Vbf16_expe_cut__pch.h"
#include "Vbf16_expe_cut__Syms.h"
#include "Vbf16_expe_cut___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_cut___024root___dump_triggers__ico(Vbf16_expe_cut___024root* vlSelf);
#endif  // VL_DEBUG

void Vbf16_expe_cut___024root___eval_triggers__ico(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_triggers__ico\n"); );
    // Body
    vlSelf->__VicoTriggered.set(0U, (IData)(vlSelf->__VicoFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vbf16_expe_cut___024root___dump_triggers__ico(vlSelf);
    }
#endif
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_cut___024root___dump_triggers__act(Vbf16_expe_cut___024root* vlSelf);
#endif  // VL_DEBUG

void Vbf16_expe_cut___024root___eval_triggers__act(Vbf16_expe_cut___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_cut__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_cut___024root___eval_triggers__act\n"); );
    // Body
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vbf16_expe_cut___024root___dump_triggers__act(vlSelf);
    }
#endif
}
