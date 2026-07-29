// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vbf16_exp2.h for the primary calling header

#include "Vbf16_exp2__pch.h"
#include "Vbf16_exp2___024root.h"

VL_INLINE_OPT void Vbf16_exp2___024root___ico_sequent__TOP__0(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___ico_sequent__TOP__0\n"); );
    // Init
    IData/*21:0*/ bf16_exp2__DOT__s1_decomposed;
    bf16_exp2__DOT__s1_decomposed = 0;
    CData/*7:0*/ bf16_exp2__DOT__s3_mant_src;
    bf16_exp2__DOT__s3_mant_src = 0;
    IData/*30:0*/ bf16_exp2__DOT__s3_mant_out;
    bf16_exp2__DOT__s3_mant_out = 0;
    CData/*3:0*/ bf16_exp2__DOT__u_decompose__DOT__status_comb;
    bf16_exp2__DOT__u_decompose__DOT__status_comb = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_decompose__DOT__exponent_comb;
    bf16_exp2__DOT__u_decompose__DOT__exponent_comb = 0;
    CData/*0:0*/ bf16_exp2__DOT__u_decompose__DOT__hidden_bit_comb;
    bf16_exp2__DOT__u_decompose__DOT__hidden_bit_comb = 0;
    CData/*1:0*/ bf16_exp2__DOT__u_early_out__DOT__eo_comb;
    bf16_exp2__DOT__u_early_out__DOT__eo_comb = 0;
    QData/*46:0*/ bf16_exp2__DOT__u_unified_shift__DOT__unified_shifted;
    bf16_exp2__DOT__u_unified_shift__DOT__unified_shifted = 0;
    QData/*41:0*/ bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff;
    bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff = 0;
    QData/*58:0*/ bf16_exp2__DOT__u_lin_approx__DOT__ax_unsigned;
    bf16_exp2__DOT__u_lin_approx__DOT__ax_unsigned = 0;
    QData/*61:0*/ bf16_exp2__DOT__u_lin_approx__DOT__neg_ax;
    bf16_exp2__DOT__u_lin_approx__DOT__neg_ax = 0;
    QData/*61:0*/ bf16_exp2__DOT__u_lin_approx__DOT__b_aligned;
    bf16_exp2__DOT__u_lin_approx__DOT__b_aligned = 0;
    QData/*61:0*/ bf16_exp2__DOT__u_lin_approx__DOT__calc_res;
    bf16_exp2__DOT__u_lin_approx__DOT__calc_res = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb;
    bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_normalize__DOT__shift_amt;
    bf16_exp2__DOT__u_normalize__DOT__shift_amt = 0;
    QData/*61:0*/ bf16_exp2__DOT__u_normalize__DOT__shifted_res;
    bf16_exp2__DOT__u_normalize__DOT__shifted_res = 0;
    QData/*58:0*/ bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb;
    bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb = 0;
    IData/*31:0*/ bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i;
    bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_round__DOT__final_exponent;
    bf16_exp2__DOT__u_round__DOT__final_exponent = 0;
    CData/*0:0*/ bf16_exp2__DOT__u_round__DOT__is_sub;
    bf16_exp2__DOT__u_round__DOT__is_sub = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_round__DOT__shift_9;
    bf16_exp2__DOT__u_round__DOT__shift_9 = 0;
    CData/*0:0*/ bf16_exp2__DOT__u_round__DOT__lsb_bit;
    bf16_exp2__DOT__u_round__DOT__lsb_bit = 0;
    CData/*0:0*/ bf16_exp2__DOT__u_round__DOT__guard_bit;
    bf16_exp2__DOT__u_round__DOT__guard_bit = 0;
    CData/*0:0*/ bf16_exp2__DOT__u_round__DOT__round_up;
    bf16_exp2__DOT__u_round__DOT__round_up = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_round__DOT__sum_m_ext;
    bf16_exp2__DOT__u_round__DOT__sum_m_ext = 0;
    QData/*58:0*/ bf16_exp2__DOT__u_round__DOT__sticky_masked;
    bf16_exp2__DOT__u_round__DOT__sticky_masked = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_round__DOT__result_m_ext;
    bf16_exp2__DOT__u_round__DOT__result_m_ext = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_round__DOT__adjusted_exp;
    bf16_exp2__DOT__u_round__DOT__adjusted_exp = 0;
    IData/*21:0*/ bf16_exp2__DOT__u_round__DOT__rounded_comb;
    bf16_exp2__DOT__u_round__DOT__rounded_comb = 0;
    SData/*15:0*/ bf16_exp2__DOT__u_recompose__DOT__bf16_comb;
    bf16_exp2__DOT__u_recompose__DOT__bf16_comb = 0;
    CData/*7:0*/ bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0;
    CData/*6:0*/ bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0;
    SData/*8:0*/ bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0;
    // Body
    vlSelf->m_axis_tvalid = vlSelf->s_axis_tvalid;
    vlSelf->s_axis_tready = vlSelf->m_axis_tready;
    bf16_exp2__DOT__u_decompose__DOT__status_comb = 
        (((IData)(((0x7f80U == (0x7f80U & (IData)(vlSelf->s_axis_tdata))) 
                   & (0U != (0x7fU & (IData)(vlSelf->s_axis_tdata))))) 
          << 3U) | (((IData)((0x7f80U == (0x7fffU & (IData)(vlSelf->s_axis_tdata)))) 
                     << 2U) | (((IData)(((0U == (0x7f80U 
                                                 & (IData)(vlSelf->s_axis_tdata))) 
                                         & (0U != (0x7fU 
                                                   & (IData)(vlSelf->s_axis_tdata))))) 
                                << 1U) | (IData)((0U 
                                                  == 
                                                  (0x7fffU 
                                                   & (IData)(vlSelf->s_axis_tdata)))))));
    bf16_exp2__DOT__u_decompose__DOT__exponent_comb 
        = ((IData)((0U != (0xdU & (IData)(bf16_exp2__DOT__u_decompose__DOT__status_comb))))
            ? 0U : ((2U & (IData)(bf16_exp2__DOT__u_decompose__DOT__status_comb))
                     ? 0x182U : (0x1ffU & ((0xffU & 
                                            ((IData)(vlSelf->s_axis_tdata) 
                                             >> 7U)) 
                                           - (IData)(0x7fU)))));
    bf16_exp2__DOT__u_decompose__DOT__hidden_bit_comb 
        = ((1U & (~ (IData)((0U != (0xdU & (IData)(bf16_exp2__DOT__u_decompose__DOT__status_comb)))))) 
           && (1U & (~ ((IData)(bf16_exp2__DOT__u_decompose__DOT__status_comb) 
                        >> 1U))));
    bf16_exp2__DOT__s1_decomposed = ((0x200000U & ((IData)(vlSelf->s_axis_tdata) 
                                                   << 6U)) 
                                     | (((IData)(bf16_exp2__DOT__u_decompose__DOT__exponent_comb) 
                                         << 0xcU) | 
                                        ((0xfe0U & 
                                          ((IData)(vlSelf->s_axis_tdata) 
                                           << 5U)) 
                                         | (((IData)(bf16_exp2__DOT__u_decompose__DOT__hidden_bit_comb) 
                                             << 4U) 
                                            | (IData)(bf16_exp2__DOT__u_decompose__DOT__status_comb)))));
    bf16_exp2__DOT__s3_mant_src = (((IData)(bf16_exp2__DOT__u_decompose__DOT__hidden_bit_comb) 
                                    << 7U) | (0x7fU 
                                              & (IData)(vlSelf->s_axis_tdata)));
    bf16_exp2__DOT__u_early_out__DOT__eo_comb = ((8U 
                                                  & bf16_exp2__DOT__s1_decomposed)
                                                  ? 3U
                                                  : 
                                                 ((1U 
                                                   & bf16_exp2__DOT__s1_decomposed)
                                                   ? 1U
                                                   : 
                                                  ((4U 
                                                    & bf16_exp2__DOT__s1_decomposed)
                                                    ? 
                                                   ((0x200000U 
                                                     & bf16_exp2__DOT__s1_decomposed)
                                                     ? 2U
                                                     : 1U)
                                                    : 
                                                   ((0x200000U 
                                                     & bf16_exp2__DOT__s1_decomposed)
                                                     ? 
                                                    (VL_GTS_III(9, 0x1f7U, 
                                                                (0x1ffU 
                                                                 & (bf16_exp2__DOT__s1_decomposed 
                                                                    >> 0xcU)))
                                                      ? 1U
                                                      : 
                                                     (VL_LTS_III(9, 7U, 
                                                                 (0x1ffU 
                                                                  & (bf16_exp2__DOT__s1_decomposed 
                                                                     >> 0xcU)))
                                                       ? 2U
                                                       : 0U))
                                                     : 1U))));
    bf16_exp2__DOT__s3_mant_out = ((IData)(vlSelf->s_axis_tuser)
                                    ? ((IData)(bf16_exp2__DOT__s3_mant_src) 
                                       << 0x16U) : 
                                   (0x7fffff00U & (
                                                   ((0x7fffffU 
                                                     & (((IData)(0x5c551dU) 
                                                         * (IData)(bf16_exp2__DOT__s3_mant_src)) 
                                                        >> 8U)) 
                                                    + 
                                                    (1U 
                                                     & ((((IData)(0x5c551dU) 
                                                          * (IData)(bf16_exp2__DOT__s3_mant_src)) 
                                                         >> 7U) 
                                                        & ((0x7fffffU 
                                                            & (((IData)(0x5c551dU) 
                                                                * (IData)(bf16_exp2__DOT__s3_mant_src)) 
                                                               >> 8U)) 
                                                           | (0U 
                                                              != 
                                                              (0x7fU 
                                                               & ((IData)(0x5c551dU) 
                                                                  * (IData)(bf16_exp2__DOT__s3_mant_src)))))))) 
                                                   << 8U)));
    bf16_exp2__DOT__u_unified_shift__DOT__unified_shifted 
        = (0x7fffffffffffULL & (VL_SHIFTL_QQI(47,47,32, (QData)((IData)(
                                                                        ((0x7fffU 
                                                                          & (bf16_exp2__DOT__s3_mant_out 
                                                                             >> 0x10U)) 
                                                                         * 
                                                                         (0x1ffffU 
                                                                          & ((IData)(1U) 
                                                                             << 
                                                                             (0x1fU 
                                                                              & ((IData)(9U) 
                                                                                + (IData)(bf16_exp2__DOT__u_decompose__DOT__exponent_comb)))))))), 0x10U) 
                                | (0x1ffffffffULL & 
                                   ((QData)((IData)(
                                                    (0xffffU 
                                                     & bf16_exp2__DOT__s3_mant_out))) 
                                    * (QData)((IData)(
                                                      (0x1ffffU 
                                                       & ((IData)(1U) 
                                                          << 
                                                          (0x1fU 
                                                           & ((IData)(9U) 
                                                              + (IData)(bf16_exp2__DOT__u_decompose__DOT__exponent_comb)))))))))));
    bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff 
        = vlSelf->bf16_exp2__DOT__u_lin_approx__DOT__u_coeff_rom__DOT__rom
        [(0x7fU & ((IData)(0x7fU) - (IData)((bf16_exp2__DOT__u_unified_shift__DOT__unified_shifted 
                                             >> 0x1fU))))];
    bf16_exp2__DOT__u_lin_approx__DOT__ax_unsigned 
        = (0x7ffffffffffffffULL & VL_SHIFTL_QQI(59,59,32, 
                                                (0x7ffffffffffffULL 
                                                 & ((QData)((IData)(
                                                                    (0x1fffffU 
                                                                     & (IData)(bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff)))) 
                                                    * (QData)((IData)(
                                                                      (0x3fffffffU 
                                                                       & (IData)(
                                                                                (bf16_exp2__DOT__u_unified_shift__DOT__unified_shifted 
                                                                                >> 8U))))))), 8U));
    bf16_exp2__DOT__u_lin_approx__DOT__neg_ax = (0x3fffffffffffffffULL 
                                                 & (- 
                                                    VL_EXTENDS_QQ(62,60, bf16_exp2__DOT__u_lin_approx__DOT__ax_unsigned)));
    bf16_exp2__DOT__u_lin_approx__DOT__b_aligned = 
        ((QData)((IData)((0x1fffffU & (IData)((bf16_exp2__DOT__u_lin_approx__DOT__packed_coeff 
                                               >> 0x15U))))) 
         << 0x26U);
    bf16_exp2__DOT__u_lin_approx__DOT__calc_res = (0x3fffffffffffffffULL 
                                                   & (bf16_exp2__DOT__u_lin_approx__DOT__b_aligned 
                                                      + bf16_exp2__DOT__u_lin_approx__DOT__neg_ax));
    bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb = 0x1ffU;
    bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i = 0x3dU;
    {
        while (VL_LTES_III(32, 0U, bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) {
            if (((0x3dU >= (0x3fU & bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)) 
                 && (1U & (IData)((bf16_exp2__DOT__u_lin_approx__DOT__calc_res 
                                   >> (0x3fU & bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i)))))) {
                bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb 
                    = (0x1ffU & bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i);
                goto __Vlabel1;
            }
            bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                = (bf16_exp2__DOT__u_normalize__DOT__unnamedblk1__DOT__i 
                   - (IData)(1U));
        }
        __Vlabel1: ;
    }
    bf16_exp2__DOT__u_normalize__DOT__shift_amt = (0x1ffU 
                                                   & ((IData)(0x3dU) 
                                                      - (IData)(bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb)));
    bf16_exp2__DOT__u_normalize__DOT__shifted_res = 
        (0x3fffffffffffffffULL & VL_SHIFTL_QQI(62,62,9, bf16_exp2__DOT__u_lin_approx__DOT__calc_res, (IData)(bf16_exp2__DOT__u_normalize__DOT__shift_amt)));
    bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb 
        = (0x7ffffffffffffffULL & (bf16_exp2__DOT__u_normalize__DOT__shifted_res 
                                   >> 3U));
    bf16_exp2__DOT__u_round__DOT__final_exponent = 
        (0x1ffU & (((IData)(bf16_exp2__DOT__u_normalize__DOT__msb_idx_comb) 
                    - (IData)(0x3aU)) + (- (IData)(
                                                   (bf16_exp2__DOT__u_unified_shift__DOT__unified_shifted 
                                                    >> 0x26U)))));
    bf16_exp2__DOT__u_round__DOT__is_sub = VL_GTS_III(9, 0x182U, (IData)(bf16_exp2__DOT__u_round__DOT__final_exponent));
    if (bf16_exp2__DOT__u_round__DOT__is_sub) {
        bf16_exp2__DOT__u_round__DOT__shift_9 = (0x1ffU 
                                                 & ((IData)(0x33U) 
                                                    + 
                                                    ((IData)(0x182U) 
                                                     - (IData)(bf16_exp2__DOT__u_round__DOT__final_exponent))));
        bf16_exp2__DOT__u_round__DOT__adjusted_exp = 0x182U;
    } else {
        bf16_exp2__DOT__u_round__DOT__shift_9 = 0x33U;
        bf16_exp2__DOT__u_round__DOT__adjusted_exp 
            = bf16_exp2__DOT__u_round__DOT__final_exponent;
    }
    bf16_exp2__DOT__u_round__DOT__sticky_masked = (bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb 
                                                   & (((QData)((IData)(
                                                                       VL_LTS_III(9, 0x3bU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                       << 0x3aU) 
                                                      | (((QData)((IData)(
                                                                          VL_LTS_III(9, 0x3aU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                          << 0x39U) 
                                                         | (((QData)((IData)(
                                                                             VL_LTS_III(9, 0x39U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                             << 0x38U) 
                                                            | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x38U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                << 0x37U) 
                                                               | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x37U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                   << 0x36U) 
                                                                  | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x36U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                      << 0x35U) 
                                                                     | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x35U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                         << 0x34U) 
                                                                        | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x34U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                            << 0x33U) 
                                                                           | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x33U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                               << 0x32U) 
                                                                              | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x32U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x31U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x31U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x30U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x30U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x2fU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2fU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x2eU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2eU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x2dU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2dU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x2cU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2cU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x2bU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2bU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x2aU) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x2aU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x29U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x29U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x28U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x28U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x27U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x27U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x26U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x26U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x25U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x25U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x24U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x24U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x23U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x23U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x22U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x22U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x21U) 
                                                                                | (((QData)((IData)(
                                                                                VL_LTS_III(9, 0x21U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))) 
                                                                                << 0x20U) 
                                                                                | (QData)((IData)(
                                                                                ((VL_LTS_III(9, 0x20U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x1fU) 
                                                                                | ((VL_LTS_III(9, 0x1fU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x1eU) 
                                                                                | ((VL_LTS_III(9, 0x1eU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x1dU) 
                                                                                | ((VL_LTS_III(9, 0x1dU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x1cU) 
                                                                                | ((VL_LTS_III(9, 0x1cU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x1bU) 
                                                                                | ((VL_LTS_III(9, 0x1bU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x1aU) 
                                                                                | ((VL_LTS_III(9, 0x1aU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x19U) 
                                                                                | ((VL_LTS_III(9, 0x19U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x18U) 
                                                                                | ((VL_LTS_III(9, 0x18U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x17U) 
                                                                                | ((VL_LTS_III(9, 0x17U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x16U) 
                                                                                | ((VL_LTS_III(9, 0x16U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x15U) 
                                                                                | ((VL_LTS_III(9, 0x15U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x14U) 
                                                                                | ((VL_LTS_III(9, 0x14U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x13U) 
                                                                                | ((VL_LTS_III(9, 0x13U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x12U) 
                                                                                | ((VL_LTS_III(9, 0x12U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x11U) 
                                                                                | ((VL_LTS_III(9, 0x11U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0x10U) 
                                                                                | ((VL_LTS_III(9, 0x10U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0xfU) 
                                                                                | ((VL_LTS_III(9, 0xfU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0xeU) 
                                                                                | ((VL_LTS_III(9, 0xeU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0xdU) 
                                                                                | ((VL_LTS_III(9, 0xdU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0xcU) 
                                                                                | ((VL_LTS_III(9, 0xcU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0xbU) 
                                                                                | ((VL_LTS_III(9, 0xbU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 0xaU) 
                                                                                | ((VL_LTS_III(9, 0xaU, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 9U) 
                                                                                | ((VL_LTS_III(9, 9U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 8U) 
                                                                                | ((VL_LTS_III(9, 8U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 7U) 
                                                                                | ((VL_LTS_III(9, 7U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 6U) 
                                                                                | ((VL_LTS_III(9, 6U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 5U) 
                                                                                | ((VL_LTS_III(9, 5U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 4U) 
                                                                                | ((VL_LTS_III(9, 4U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 3U) 
                                                                                | ((VL_LTS_III(9, 3U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 2U) 
                                                                                | ((VL_LTS_III(9, 2U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                                << 1U) 
                                                                                | VL_LTS_III(9, 1U, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));
    bf16_exp2__DOT__u_round__DOT__lsb_bit = (VL_GTS_III(32, 0x3bU, 
                                                        VL_EXTENDS_II(32,9, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9))) 
                                             & ((0x3aU 
                                                 >= 
                                                 (0x3fU 
                                                  & (IData)(bf16_exp2__DOT__u_round__DOT__shift_9))) 
                                                && (1U 
                                                    & (IData)(
                                                              (bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb 
                                                               >> 
                                                               (0x3fU 
                                                                & (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))))));
    bf16_exp2__DOT__u_round__DOT__guard_bit = (VL_LTS_III(32, 0U, 
                                                          VL_EXTENDS_II(32,9, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9))) 
                                               & (VL_GTES_III(32, 0x3bU, 
                                                              VL_EXTENDS_II(32,9, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9))) 
                                                  & ((0x3aU 
                                                      >= 
                                                      (0x3fU 
                                                       & (VL_EXTENDS_II(6,9, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                          - (IData)(1U)))) 
                                                     && (1U 
                                                         & (IData)(
                                                                   (bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb 
                                                                    >> 
                                                                    (0x3fU 
                                                                     & (VL_EXTENDS_II(6,9, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)) 
                                                                        - (IData)(1U)))))))));
    bf16_exp2__DOT__u_round__DOT__round_up = ((IData)(bf16_exp2__DOT__u_round__DOT__guard_bit) 
                                              & ((IData)(bf16_exp2__DOT__u_round__DOT__lsb_bit) 
                                                 | (IData)(
                                                           (0ULL 
                                                            != bf16_exp2__DOT__u_round__DOT__sticky_masked))));
    bf16_exp2__DOT__u_round__DOT__sum_m_ext = (VL_GTS_III(32, 0x3bU, 
                                                          VL_EXTENDS_II(32,9, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))
                                                ? (0x1ffU 
                                                   & (IData)(
                                                             (0x7ffffffffffffffULL 
                                                              & VL_SHIFTR_QQI(59,59,9, bf16_exp2__DOT__u_normalize__DOT__poly_mant_comb, (IData)(bf16_exp2__DOT__u_round__DOT__shift_9)))))
                                                : 0U);
    if (bf16_exp2__DOT__u_round__DOT__round_up) {
        bf16_exp2__DOT__u_round__DOT__sum_m_ext = (0x1ffU 
                                                   & ((IData)(1U) 
                                                      + (IData)(bf16_exp2__DOT__u_round__DOT__sum_m_ext)));
    }
    bf16_exp2__DOT__u_round__DOT__result_m_ext = bf16_exp2__DOT__u_round__DOT__sum_m_ext;
    if ((0x100U & (IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext))) {
        bf16_exp2__DOT__u_round__DOT__adjusted_exp 
            = (0x1ffU & ((IData)(1U) + (IData)(bf16_exp2__DOT__u_round__DOT__adjusted_exp)));
        bf16_exp2__DOT__u_round__DOT__result_m_ext 
            = (0x1ffU & VL_SHIFTR_III(9,9,32, (IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext), 1U));
    }
    bf16_exp2__DOT__u_round__DOT__rounded_comb = 0U;
    bf16_exp2__DOT__u_round__DOT__rounded_comb = (0x1fffffU 
                                                  & bf16_exp2__DOT__u_round__DOT__rounded_comb);
    if ((0U == (IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext))) {
        bf16_exp2__DOT__u_round__DOT__rounded_comb 
            = (1U | bf16_exp2__DOT__u_round__DOT__rounded_comb);
    } else if (((IData)(bf16_exp2__DOT__u_round__DOT__is_sub) 
                & (~ ((IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext) 
                      >> 7U)))) {
        bf16_exp2__DOT__u_round__DOT__rounded_comb 
            = ((0x20000fU & bf16_exp2__DOT__u_round__DOT__rounded_comb) 
               | (0x181000U | (0xfe0U & ((IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext) 
                                         << 5U))));
        bf16_exp2__DOT__u_round__DOT__rounded_comb 
            = (2U | bf16_exp2__DOT__u_round__DOT__rounded_comb);
    } else {
        bf16_exp2__DOT__u_round__DOT__rounded_comb 
            = ((0x20000fU & bf16_exp2__DOT__u_round__DOT__rounded_comb) 
               | (0x10U | (((IData)(bf16_exp2__DOT__u_round__DOT__adjusted_exp) 
                            << 0xcU) | (0xfe0U & ((IData)(bf16_exp2__DOT__u_round__DOT__result_m_ext) 
                                                  << 5U)))));
    }
    if ((3U == (IData)(bf16_exp2__DOT__u_early_out__DOT__eo_comb))) {
        vlSelf->bf16_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_exp2__DOT__s8_final_fp = (8U | vlSelf->bf16_exp2__DOT__s8_final_fp);
        vlSelf->bf16_exp2__DOT__s8_final_fp = (0x200000U 
                                               | vlSelf->bf16_exp2__DOT__s8_final_fp);
        vlSelf->bf16_exp2__DOT__s8_final_fp = (0x800U 
                                               | (0x3ff01fU 
                                                  & vlSelf->bf16_exp2__DOT__s8_final_fp));
    } else if ((1U == (IData)(bf16_exp2__DOT__u_early_out__DOT__eo_comb))) {
        vlSelf->bf16_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_exp2__DOT__s8_final_fp = (0x200fffU 
                                               & vlSelf->bf16_exp2__DOT__s8_final_fp);
        vlSelf->bf16_exp2__DOT__s8_final_fp = (0x10U 
                                               | vlSelf->bf16_exp2__DOT__s8_final_fp);
    } else if ((2U == (IData)(bf16_exp2__DOT__u_early_out__DOT__eo_comb))) {
        vlSelf->bf16_exp2__DOT__s8_final_fp = 0U;
        vlSelf->bf16_exp2__DOT__s8_final_fp = (1U | vlSelf->bf16_exp2__DOT__s8_final_fp);
    } else {
        vlSelf->bf16_exp2__DOT__s8_final_fp = bf16_exp2__DOT__u_round__DOT__rounded_comb;
    }
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
    bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp = 0U;
    if (VL_ONEHOT0_I(((8U & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                             << 2U)) | ((4U & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                               >> 1U)) 
                                        | ((2U & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                                  >> 1U)) 
                                           | (1U & vlSelf->bf16_exp2__DOT__s8_final_fp)))))) {
        if ((1U & vlSelf->bf16_exp2__DOT__s8_final_fp)) {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
        } else if ((4U & vlSelf->bf16_exp2__DOT__s8_final_fp)) {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
        } else if ((8U & vlSelf->bf16_exp2__DOT__s8_final_fp)) {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                = ((0U == (0x7fU & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                    >> 5U))) ? 0x40U
                    : (0x7fU & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                >> 5U)));
        } else if ((2U & vlSelf->bf16_exp2__DOT__s8_final_fp)) {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                = (0x7fU & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                            >> 5U));
        } else {
            bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp 
                = (0x1ffU & ((IData)(0x7fU) + (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                               >> 0xcU)));
            if (VL_GTES_III(9, 0U, (IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp))) {
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0U;
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
            } else if (VL_LTES_III(9, 0xffU, (IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp))) {
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp = 0xffU;
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out = 0U;
            } else {
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp 
                    = (0xffU & (IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__temp_exp));
                bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out 
                    = (0x7fU & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                >> 5U));
            }
        }
    }
    bf16_exp2__DOT__u_recompose__DOT__bf16_comb = (
                                                   (0x8000U 
                                                    & (vlSelf->bf16_exp2__DOT__s8_final_fp 
                                                       >> 6U)) 
                                                   | (((IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__biased_exp) 
                                                       << 7U) 
                                                      | (IData)(bf16_exp2__DOT__u_recompose__DOT__unnamedblk1__DOT__mant_out)));
    vlSelf->m_axis_tdata = bf16_exp2__DOT__u_recompose__DOT__bf16_comb;
}

void Vbf16_exp2___024root___eval_ico(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vbf16_exp2___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vbf16_exp2___024root___eval_triggers__ico(Vbf16_exp2___024root* vlSelf);

bool Vbf16_exp2___024root___eval_phase__ico(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vbf16_exp2___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vbf16_exp2___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vbf16_exp2___024root___eval_act(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_act\n"); );
}

void Vbf16_exp2___024root___eval_nba(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_nba\n"); );
}

void Vbf16_exp2___024root___eval_triggers__act(Vbf16_exp2___024root* vlSelf);

bool Vbf16_exp2___024root___eval_phase__act(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<0> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vbf16_exp2___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vbf16_exp2___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vbf16_exp2___024root___eval_phase__nba(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vbf16_exp2___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_exp2___024root___dump_triggers__ico(Vbf16_exp2___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_exp2___024root___dump_triggers__nba(Vbf16_exp2___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vbf16_exp2___024root___dump_triggers__act(Vbf16_exp2___024root* vlSelf);
#endif  // VL_DEBUG

void Vbf16_exp2___024root___eval(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VicoIterCount;
    CData/*0:0*/ __VicoContinue;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VicoIterCount = 0U;
    vlSelf->__VicoFirstIteration = 1U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        if (VL_UNLIKELY((0x64U < __VicoIterCount))) {
#ifdef VL_DEBUG
            Vbf16_exp2___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_exp2.sv", 30, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vbf16_exp2___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vbf16_exp2___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("src/rtl/bf16_exp2.sv", 30, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vbf16_exp2___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("src/rtl/bf16_exp2.sv", 30, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vbf16_exp2___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vbf16_exp2___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vbf16_exp2___024root___eval_debug_assertions(Vbf16_exp2___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vbf16_exp2__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vbf16_exp2___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->rst_n & 0xfeU))) {
        Verilated::overWidthError("rst_n");}
    if (VL_UNLIKELY((vlSelf->s_axis_tuser & 0xfeU))) {
        Verilated::overWidthError("s_axis_tuser");}
    if (VL_UNLIKELY((vlSelf->s_axis_tvalid & 0xfeU))) {
        Verilated::overWidthError("s_axis_tvalid");}
    if (VL_UNLIKELY((vlSelf->m_axis_tready & 0xfeU))) {
        Verilated::overWidthError("m_axis_tready");}
}
#endif  // VL_DEBUG
