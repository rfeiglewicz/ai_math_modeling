#include <iostream>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Total number of possible values for a 16-bit type (2^16)
const int TOTAL_VALUES = 65536;

// CUDA Kernel to calculate 2^x for every possible bf16 bit pattern
__global__ void generate_exp2_bf16_kernel(unsigned short* output_raw) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not go out of bounds
    if (idx < TOTAL_VALUES) {
        // 1. Interpret the index (0-65535) as the raw bits of a bfloat16 input
        unsigned short input_bits = static_cast<unsigned short>(idx);
        
        // Use dedicated CUDA intrinsic to reinterpret bits as __nv_bfloat16
        __nv_bfloat16 input_val = __ushort_as_bfloat16(input_bits);

        // 2. Calculate 2^x
        // Note: Standard math operations for bf16 often involve conversion to float 
        // to maintain precision before casting back, as direct h2exp2 is for half2.
        // We use dedicated conversion intrinsics here.
        float input_f = __bfloat162float(input_val);
        float result_f = exp2f(input_f);
        
        // Convert result back to bfloat16 using dedicated intrinsic
        __nv_bfloat16 result_val = __float2bfloat16(result_f);

        // 3. Store the raw bits of the result
        output_raw[idx] = __bfloat16_as_ushort(result_val);
    }
}

int main() {
    // Host memory allocation
    size_t size_bytes = TOTAL_VALUES * sizeof(unsigned short);
    unsigned short* h_output = (unsigned short*)malloc(size_bytes);

    // Device memory allocation
    unsigned short* d_output;
    cudaError_t err = cudaMalloc(&d_output, size_bytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Kernel Configuration
    // We need 65536 threads total. 
    int threadsPerBlock = 256;
    int blocksPerGrid = (TOTAL_VALUES + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching CUDA kernel..." << std::endl;
    
    // Launch Kernel
    generate_exp2_bf16_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output);

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output);
        free(h_output);
        return -1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_output, d_output, size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Memcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_output);
        free(h_output);
        return -1;
    }

    std::cout << "Calculation complete. Writing to output.txt..." << std::endl;

    // Write to file
    std::ofstream outfile("output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        cudaFree(d_output);
        free(h_output);
        return -1;
    }

    // Configure output format: Hex, lowercase, 4 digits, zero padding
    outfile << std::hex << std::setfill('0');

    for (int i = 0; i < TOTAL_VALUES; ++i) {
        // i represents the input hex (since we mapped index to bit pattern)
        // h_output[i] represents the result hex
        outfile << std::setw(4) << i << " " << std::setw(4) << h_output[i] << "\n";
    }

    outfile.close();
    std::cout << "Done! Results saved to output.txt" << std::endl;

    // Cleanup
    cudaFree(d_output);
    free(h_output);

    return 0;
}