# Hybrid compressed-table BF16 `exp(x)`

This implementation is bit-exact with the full-LUT model (`bf16_expe_lut_approx`),
but it uses two representations selected by the unbiased input exponent.

| Exponent range | Representation | Stored payload |
|---|---|---:|
| `[-9, -2]` | Initial BF16 code plus mantissa transition thresholds | 98 × 7 + 8 × 16 = 814 bits |
| `[-1, 6]` | Direct BF16 ROM | 1024 × 16 = 16,384 bits |
| `7` | Early-out to +0.0 (`bf16_early_out`, `MAX_EXP = 6`) | 0 bits |

Exponent 7 (x <= -128) always gives +0.0, so the RTL handles it in the
early-out stage, like inputs beyond the table. The datapath therefore covers
`[-9, 6]` only, and the route needs a single bit: with `idx = exp + 9` in
`[0, 15]`, `idx[3]` selects dense vs sparse and `idx[2:0]` is the bin index of
both halves.

The sparse encoding is possible because every output transition in exponent
bins `[-9, -2]` decrements the unsigned BF16 payload by exactly one. The
mantissa thresholds therefore encode the complete output sequence without
storing each 16-bit output.

## Memory comparison

- Full LUT payload (C++ table, exponents `[-9, 7]`): 34,816 bits; the RTL
  full-LUT ROM drops the all-zero exponent-7 row and stores 2048 × 16 =
  32,768 bits
- Hybrid payload: 17,198 bits
- Logical payload reduction vs the C++ table: 50.6032%

On an FPGA, the dense part fits in one 18-Kbit block RAM, while the full-LUT
ROM takes one 36-Kbit block RAM (2048 × 16). The sparse threshold
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
