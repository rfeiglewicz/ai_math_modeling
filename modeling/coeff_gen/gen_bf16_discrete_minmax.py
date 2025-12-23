"""
Optimal exp2(x) Linear Approximation Coefficient Generator for BF16

This script generates coefficients (a, b) for piecewise linear approximation:
    y_approx = a * x + b

The optimization minimizes maximum ULP error for BF16 format by:
1. Enumerating all discrete BF16 values in each interval
2. Using minimax optimization to find optimal coefficients
3. Evaluating error in ULP units for BF16 precision

Author: Generated for ai_math_modeling project
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from typing import Tuple, List, NamedTuple
import struct
import argparse
from datetime import datetime


class LUTEntry(NamedTuple):
    """Single LUT entry with coefficients and error metrics."""
    index: int
    x_start: float
    x_end: float
    slope_a: float
    intercept_b: float
    max_ulp_error: float
    avg_ulp_error: float


def float32_to_bits(f: float) -> int:
    """Convert float32 to its bit representation."""
    return struct.unpack('!I', struct.pack('!f', f))[0]


def bits_to_float32(bits: int) -> float:
    """Convert bit representation to float32."""
    return struct.unpack('!f', struct.pack('!I', bits))[0]


def get_bf16_values_in_range(start: float, end: float) -> np.ndarray:
    """
    Generate all representable BF16 values in range [start, end].
    Both boundaries are INCLUDED.
    
    BF16 uses the upper 16 bits of float32.
    """
    bf16_step = 1 << 16  # BF16 increment in 32-bit space
    values = []
    
    if start >= 0 and end > 0:
        # Positive range [start, end]
        start_bits = float32_to_bits(start)
        # Align to BF16 (round up to ensure we start >= start)
        start_bits_aligned = (start_bits + 0xFFFF) & 0xFFFF0000
        
        end_bits = float32_to_bits(end)
        
        current_bits = start_bits_aligned
        # Iterate while bits correspond to values <= end
        while current_bits <= end_bits + bf16_step: # Safety margin
            val = bits_to_float32(current_bits)
            if val > end:
                break
            if val >= start:
                values.append(val)
            
            if current_bits > 0xFFFFFFFF - bf16_step:
                break
            current_bits += bf16_step
            
    elif start < 0 and end <= 0:
        # Negative range [start, end]
        start_bits = float32_to_bits(start) 
        end_bits = float32_to_bits(end)     
        
        # Align end_bits (start of iteration)
        current_bits = end_bits & 0xFFFF0000
        
        while current_bits <= start_bits + bf16_step:
            val = bits_to_float32(current_bits)
            
            if val < start:
                break
                
            if val <= end:
                values.append(val)
                
            if current_bits > 0xFFFFFFFF - bf16_step:
                break
            current_bits += bf16_step
            
    elif start < 0 and end > 0:
        # Mixed range
        neg_vals = get_bf16_values_in_range(start, -0.0)
        pos_vals = get_bf16_values_in_range(0.0, end)
        combined = np.concatenate([neg_vals, pos_vals])
        return np.unique(combined)
    
    # Sort by actual value
    values.sort()
    return np.array(values, dtype=np.float64)


def calculate_ulp_bf16(value: float) -> float:
    """
    Calculate the size of 1 ULP for a given value in BF16 format.
    
    For BF16: 7 mantissa bits, so 1 ULP = 2^(exponent - 7)
    """
    if value == 0.0:
        # Minimum ULP in denormal range
        return 2.0 ** (-126 - 7)
    
    # Get exponent of the value
    exponent = int(np.floor(np.log2(np.abs(value))))
    
    # Clamp to minimum normal exponent for BF16
    min_exp = -126  # 1 - 127 (bias)
    effective_exp = max(exponent, min_exp)
    
    # 1 ULP = 2^(exponent - mantissa_bits)
    return 2.0 ** (effective_exp - 7)


def compute_ulp_errors(x_vals: np.ndarray, y_true: np.ndarray, 
                       a: float, b: float) -> np.ndarray:
    """
    Compute ULP errors for linear approximation y = a*x + b.
    
    ULP error is computed at the OUTPUT value (y_true), as that's where
    the precision matters for BF16 representation.
    """
    y_approx = a * x_vals + b
    
    # Round to BF16 using numpy bit manipulation (RNE)
    y_f32 = y_approx.astype(np.float32)
    y_bits = y_f32.view(np.uint32)
    
    # RNE Rounding Logic
    lsb = (y_bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    y_bits_rounded = y_bits + rounding_bias
    
    # Mask to keep upper 16 bits
    y_bits_rounded &= 0xFFFF0000
    
    # View back as float32 (this is our BF16 value)
    y_bf16 = y_bits_rounded.view(np.float32)
    
    abs_errors = np.abs(y_bf16 - y_true)
    
    # Calculate ULP size for each output value
    ulp_sizes = np.array([calculate_ulp_bf16(y) for y in y_true])
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = abs_errors / ulp_sizes
        result[ulp_sizes == 0] = 0 # Should not happen for exp2
    
    return result


def fit_minimax_ulp(x_points: np.ndarray, y_points: np.ndarray) -> Tuple[float, float, float]:
    """
    Find coefficients (a, b) that minimize maximum ULP error.
    """
    
    def max_ulp_error(coeffs):
        a, b = coeffs
        ulp_errors = compute_ulp_errors(x_points, y_points, a, b)
        return np.max(ulp_errors)
    
    # Initial guess using least squares
    A = np.vstack([x_points, np.ones_like(x_points)]).T
    initial_coeffs, _, _, _ = np.linalg.lstsq(A, y_points, rcond=None)
    
    # Refine using Nelder-Mead (robust for minimax problems)
    result = minimize(
        max_ulp_error, 
        initial_coeffs, 
        method='Nelder-Mead',
        options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000}
    )
    
    # Further refinement using Powell method
    result2 = minimize(
        max_ulp_error,
        result.x,
        method='Powell',
        options={'xtol': 1e-14, 'ftol': 1e-14, 'maxiter': 10000}
    )
    
    # Choose the better result
    if max_ulp_error(result2.x) < max_ulp_error(result.x):
        final_coeffs = result2.x
    else:
        final_coeffs = result.x
    
    a, b = final_coeffs
    final_max_ulp = max_ulp_error(final_coeffs)
    
    return a, b, final_max_ulp


def generate_lut_entries(interval_start: float, interval_end: float, 
                         num_entries: int) -> List[LUTEntry]:
    """
    Generate LUT entries for the given interval.
    
    Modified Logic:
    Divides the interval into equal GEOMETRIC widths to match the C++ 
    linear index calculation logic (get_lut_index).
    """
    # Get all BF16 values in the full interval for potential inclusion
    all_x = get_bf16_values_in_range(interval_start, interval_end)
    total_points = len(all_x)
    
    print(f"Interval: [{interval_start}, {interval_end})")
    print(f"Total BF16 values in range: {total_points}")
    print(f"LUT entries: {num_entries}")
    
    if total_points == 0:
        raise ValueError("No BF16 values found in the specified range.")
    
    entries = []
    
    # Calculate the fixed geometric step size
    interval_width = interval_end - interval_start
    step = interval_width / num_entries
    
    print(f"Geometric bin width: {step:.6f}")
    print()

    for i in range(num_entries):
        # Calculate geometric boundaries for the current bin
        bin_start_geo = interval_start + i * step
        bin_end_geo = interval_start + (i + 1) * step
        
        # Filter points from all_x that fall into this bin.
        # The C++ index logic is: idx = (x - start) / (end - start) * entries
        # So bin 'i' covers [bin_start_geo, bin_end_geo).
        # For the last bin, we must inclusive of the upper bound.
        
        if i == num_entries - 1:
            # Last bin: include everything up to interval_end (inclusive)
            mask = (all_x >= bin_start_geo) & (all_x <= interval_end + 1e-9)
        else:
            # Standard bin: [start, end)
            mask = (all_x >= bin_start_geo) & (all_x < bin_end_geo)
            
        x_subset = all_x[mask]
        
        # Handle cases where no BF16 values exist in a geometric bin
        if len(x_subset) == 0:
            # This can happen if the bin is extremely narrow (smaller than BF16 epsilon)
            # or in gaps between BF16 values.
            # We use the geometric midpoint to generate a 'safe' coefficient 
            # (though it won't be used for any actual value, it prevents empty LUTs).
            midpoint = (bin_start_geo + bin_end_geo) / 2.0
            x_subset = np.array([midpoint], dtype=np.float64)
            # print(f"  Notice: Bin {i} is empty of BF16 values. Using midpoint {midpoint:.6f} for coeffs.")

        y_target = np.exp2(x_subset)
        
        # Determine effective range for reporting
        x_start_actual = float(x_subset[0])
        x_end_actual = float(x_subset[-1])
        
        # Find optimal coefficients
        a, b, max_ulp = fit_minimax_ulp(x_subset, y_target)
        
        # Compute average ULP error
        ulp_errors = compute_ulp_errors(x_subset, y_target, a, b)
        avg_ulp = float(np.mean(ulp_errors))
        
        entry = LUTEntry(
            index=i,
            x_start=x_start_actual,
            x_end=x_end_actual,
            slope_a=a,
            intercept_b=b,
            max_ulp_error=max_ulp,
            avg_ulp_error=avg_ulp
        )
        entries.append(entry)
        
        if (i + 1) % max(1, num_entries // 10) == 0 or i == num_entries - 1:
            print(f"  Entry {i+1:3d}/{num_entries}: Geo=[{bin_start_geo:.5f}, {bin_end_geo:.5f}] "
                  f"max_ulp={max_ulp:.4f}")
    
    return entries


def write_hpp_file(entries: List[LUTEntry], 
                   interval_start: float, 
                   interval_end: float,
                   output_path: str):
    """Write the coefficient LUT to a C++ header file."""
    
    num_entries = len(entries)
    worst_ulp = max(e.max_ulp_error for e in entries)
    avg_ulp = sum(e.avg_ulp_error for e in entries) / num_entries
    
    with open(output_path, 'w') as f:
        f.write(f"#ifndef BF16_EXP2_COEFFS_HPP\n")
        f.write(f"#define BF16_EXP2_COEFFS_HPP\n\n")
        
        f.write(f"#include <cstdint>\n\n")
        
        f.write(f"// =============================================================\n")
        f.write(f"// exp2(x) Linear Approximation Coefficients for BF16\n")
        f.write(f"// =============================================================\n")
        f.write(f"//\n")
        f.write(f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Interval:  [{interval_start}, {interval_end}]\n")
        f.write(f"// LUT Size:  {num_entries}\n")
        f.write(f"//\n")
        f.write(f"// Approximation: y = a * x + b\n")
        f.write(f"//\n")
        f.write(f"// Error Metrics (BF16 ULP):\n")
        f.write(f"//   Worst-case: {worst_ulp:.6f} ULP\n")
        f.write(f"//   Average:    {avg_ulp:.6f} ULP\n")
        f.write(f"// =============================================================\n\n")
        
        f.write(f"namespace bf16_exp2 {{\n\n")
        
        f.write(f"constexpr int LUT_SIZE = {num_entries};\n")
        f.write(f"constexpr float INTERVAL_START = {interval_start}f;\n")
        f.write(f"constexpr float INTERVAL_END = {interval_end}f;\n\n")
        
        # Write slope coefficients (a)
        f.write(f"// Slope coefficients (a)\n")
        f.write(f"static const float coeffs_a[LUT_SIZE] = {{\n")
        for i, entry in enumerate(entries):
            comma = "," if i < num_entries - 1 else ""
            f.write(f"    {entry.slope_a: .10e}f{comma}  "
                    f"// Idx {i} ulp={entry.max_ulp_error:.4f}\n")
        f.write(f"}};\n\n")
        
        # Write intercept coefficients (b)
        f.write(f"// Intercept coefficients (b)\n")
        f.write(f"static const float coeffs_b[LUT_SIZE] = {{\n")
        for i, entry in enumerate(entries):
            comma = "," if i < num_entries - 1 else ""
            f.write(f"    {entry.intercept_b: .10e}f{comma}\n")
        f.write(f"}};\n\n")
        
        # Write a helper function for index calculation
        f.write(f"/**\n")
        f.write(f" * @brief Calculate LUT index from input value.\n")
        f.write(f" * \n")
        f.write(f" * @param x Input value in [{interval_start}, {interval_end}]\n")
        f.write(f" * @return int LUT index [0, {num_entries - 1}]\n")
        f.write(f" */\n")
        f.write(f"inline int get_lut_index(float x) {{\n")
        f.write(f"    float normalized = (x - INTERVAL_START) / (INTERVAL_END - INTERVAL_START);\n")
        f.write(f"    int idx = static_cast<int>(normalized * LUT_SIZE);\n")
        f.write(f"    // Clamp to valid range\n")
        f.write(f"    if (idx < 0) idx = 0;\n")
        f.write(f"    if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;\n")
        f.write(f"    return idx;\n")
        f.write(f"}}\n\n")
        
        f.write(f"/**\n")
        f.write(f" * @brief Compute exp2(x) approximation using LUT.\n")
        f.write(f" * \n")
        f.write(f" * @param x Input value in [{interval_start}, {interval_end}]\n")
        f.write(f" * @return float Approximation of 2^x\n")
        f.write(f" */\n")
        f.write(f"inline float exp2_approx(float x) {{\n")
        f.write(f"    int idx = get_lut_index(x);\n")
        f.write(f"    return coeffs_a[idx] * x + coeffs_b[idx];\n")
        f.write(f"}}\n\n")
        
        f.write(f"}} // namespace bf16_exp2\n\n")
        f.write(f"#endif // BF16_EXP2_COEFFS_HPP\n")
    
    print(f"\nOutput written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimal exp2(x) LUT coefficients for BF16"
    )
    parser.add_argument(
        "--start", type=float, default=0.25,
        help="Interval start (included)"
    )
    parser.add_argument(
        "--end", type=float, default=0.5,
        help="Interval end (included)"
    )
    parser.add_argument(
        "--entries", type=int, default=16,
        help="Number of LUT entries (default: 16)"
    )
    parser.add_argument(
        "--output", type=str, default="bf16_exp2_coeffs.hpp",
        help="Output header file path"
    )
    
    args = parser.parse_args()
    
    # Normalize: ensure start < end numerically
    interval_start = min(args.start, args.end)
    interval_end = max(args.start, args.end)
    
    print("=" * 60)
    print("exp2(x) LUT Coefficient Generator for BF16")
    print("=" * 60)
    print()
    
    print(f"Interval: [{interval_start}, {interval_end}] (Closed interval)")
    print()
    
    # Get BF16 values and show debug info
    all_x = get_bf16_values_in_range(interval_start, interval_end)
    print(f"Total BF16 values found: {len(all_x)}")
    if len(all_x) > 0:
        print(f"  First value: {all_x[0]:.10f}")
        print(f"  Last value:  {all_x[-1]:.10f}")
        
        # Verify boundaries
        if abs(all_x[0] - interval_start) > 1e-6:
             print(f"  NOTE: Start value {all_x[0]} differs from requested {interval_start} (alignment)")
        if abs(all_x[-1] - interval_end) > 1e-6:
             print(f"  NOTE: End value {all_x[-1]} differs from requested {interval_end} (alignment)")

    print()
    
    # Generate LUT entries
    entries = generate_lut_entries(interval_start, interval_end, args.entries)
    
    # Summary statistics
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    worst_ulp = max(e.max_ulp_error for e in entries)
    avg_max_ulp = sum(e.max_ulp_error for e in entries) / len(entries)
    print(f"Worst-case ULP error: {worst_ulp:.6f}")
    print(f"Average max ULP:      {avg_max_ulp:.6f}")
    
    # Write output file
    write_hpp_file(entries, interval_start, interval_end, args.output)


if __name__ == "__main__":
    main()