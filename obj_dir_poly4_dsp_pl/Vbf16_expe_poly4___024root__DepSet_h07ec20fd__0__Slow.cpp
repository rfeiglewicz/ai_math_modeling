// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_expe_poly4.h for the primary calling header

#include "Vbf16_expe_poly4__pch.h"
#include "Vbf16_expe_poly4__Syms.h"
#include "Vbf16_expe_poly4___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_expe_poly4___024root___dump_triggers__stl(Vbf16_expe_poly4___024root* vlSelf);
#endif  // VL_DEBUG

VL_ATTR_COLD void Vbf16_expe_poly4___024root___eval_triggers__stl(Vbf16_expe_poly4___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_expe_poly4__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_expe_poly4___024root___eval_triggers__stl\n"); );
    // Body
    vlSelf->__VstlTriggered.set(0U, (IData)(vlSelf->__VstlFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vbf16_expe_poly4___024root___dump_triggers__stl(vlSelf);
    }
#endif
}
