// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VBF16_EXPE_CUT__SYMS_H_
#define VERILATED_VBF16_EXPE_CUT__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vbf16_expe_cut.h"

// INCLUDE MODULE CLASSES
#include "Vbf16_expe_cut___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vbf16_expe_cut__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vbf16_expe_cut* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vbf16_expe_cut___024root       TOP;

    // CONSTRUCTORS
    Vbf16_expe_cut__Syms(VerilatedContext* contextp, const char* namep, Vbf16_expe_cut* modelp);
    ~Vbf16_expe_cut__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
