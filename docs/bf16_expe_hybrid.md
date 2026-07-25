# Hybrid compressed-table BF16 `exp(x)`

This implementation is bit-exact with the full 2176-entry lookup table, but it
uses two representations selected by the unbiased input exponent.

| Exponent range | Representation | Stored payload |
|---|---|---:|
| `[-9, -2]` | Initial BF16 code plus mantissa transition thresholds | 98 × 7 + 8 × 16 = 814 bits |
| `[-1, 6]` | Direct BF16 ROM | 1024 × 16 = 16,384 bits |
| `7` | Constant zero | 0 bits |

The sparse encoding is possible because every output transition in exponent
bins `[-9, -2]` decrements the unsigned BF16 payload by exactly one. The
mantissa thresholds therefore encode the complete output sequence without
storing each 16-bit output.

## Memory comparison

- Full LUT payload: 34,816 bits
- Hybrid payload: 17,198 bits
- Logical payload reduction: 50.6032%

On an FPGA, the dense part can fit in one 18-Kbit block RAM instead of the full
implementation's approximately one 36-Kbit block RAM. The sparse threshold
decoder is implemented in logic, so final LUT/ALM, timing, and power results
must be obtained from synthesis for the target FPGA. Logical payload reduction
alone is not a complete area result.

## Pipeline

With `REGISTER_STAGES=1`, the AXI-Stream implementation has four stages:

1. BF16 decomposition
2. Early-out, route selection, and address generation
3. Sparse threshold decode / dense ROM read
4. Output selection

All registers use the common `pipe_en` signal and freeze under output
backpressure.

## Verification

- `make expe_hybrid_test`: exhaustive C++ equivalence against the full-LUT model
- `make rtl_expe_hybrid_verify`: exhaustive combinational Verilator test
- `make rtl_expe_hybrid_verify_pipelined`: exhaustive four-stage Verilator test
- `make gen_expe_hybrid_tables`: regenerate the C++ table, dense RTL ROM, and
  sparse RTL decoder from the full-LUT source table

All three exhaustive tests cover all 65,536 BF16 input payloads.
