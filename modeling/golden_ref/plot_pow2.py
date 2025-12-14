import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def read_and_plot(filename='output.txt'):
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found. Please run the CUDA program first.")
        sys.exit(1)

    print(f"Reading data from {filename}...")
    
    # Load data as strings
    try:
        data = np.loadtxt(filename, dtype=str)
    except ValueError:
        print("Error: Could not parse the file. Ensure it contains two columns of hex values.")
        sys.exit(1)

    # Split columns
    x_hex = data[:, 0]
    y_hex = data[:, 1]

    print("Converting bf16 hex to float32...")

    # Helper function for conversion:
    # 1. Convert hex string to uint32
    # 2. Shift left by 16 bits (bf16 corresponds to the upper 16 bits of float32)
    # 3. Reinterpret the binary data as float32
    def bf16_hex_to_float(hex_array):
        # Convert hex strings to integers
        uint_vals = np.array([int(h, 16) for h in hex_array], dtype=np.uint32)
        # Shift bits to align with float32 format
        uint_vals = uint_vals << 16
        # View memory as float32
        return uint_vals.view(np.float32)

    # Convert both columns
    x_vals = bf16_hex_to_float(x_hex)
    y_vals = bf16_hex_to_float(y_hex)

    # Filter out NaNs and Infinities
    # bf16 has a limited range, so 2^x will overflow quickly
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    
    x_clean = x_vals[mask]
    y_clean = y_vals[mask]
    print(x_clean[16385])
    print(y_clean[16385])

    print(f"Valid data points (excluding NaN/Inf): {len(x_clean)} out of {len(x_vals)}")

    # Sort the data
    # The input file iterates bits 0x0000 -> 0xFFFF. In IEEE 754, this order is:
    # Positive Denormals -> Positive Normals -> Positive Inf -> Negative Denormals -> ...
    # We must sort by X value to draw a continuous line.
    sort_indices = np.argsort(x_clean)
    x_sorted = x_clean[sort_indices]
    y_sorted = y_clean[sort_indices]

    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.plot(x_sorted, y_sorted, '.', markersize=1, label=r'$f(x) = 2^x$ (bf16)')
    
    plt.title('Function $2^x$ in bfloat16 format')
    plt.xlabel('x (Input)')
    plt.ylabel('y (Output)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-1, 20)
    
    # Optional: Log scale to see the dynamic range better
    # plt.yscale('log') 

    output_img = 'plot_bf16.png'
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to: {output_img}")
    plt.show()

if __name__ == "__main__":
    read_and_plot()