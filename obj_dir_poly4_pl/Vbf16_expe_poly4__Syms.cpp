// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vbf16_expe_poly4__pch.h"
#include "Vbf16_expe_poly4.h"
#include "Vbf16_expe_poly4___024root.h"

// FUNCTIONS
Vbf16_expe_poly4__Syms::~Vbf16_expe_poly4__Syms()
{
}

Vbf16_expe_poly4__Syms::Vbf16_expe_poly4__Syms(VerilatedContext* contextp, const char* namep, Vbf16_expe_poly4* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
{
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-12);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
}
