// =============================================================================
// bf16_coeff_rom.sv
// 128-entry x 42-bit packed coefficient ROM for bf16_linear_approx.
//
// Each entry stores two 21-bit 1.20 coefficients packed as:
//   data[20:0]  = a  (slope)
//   data[41:21] = b  (intercept)
//
// Synthesis attribute (* rom_style = "block" *) hints Vivado to infer BRAM.
// For Intel/Quartus the equivalent is (* ramstyle = "M20K" *) -- change as needed.
//
// Parameters:
//   REGISTERED  - 1: synchronous output register (BRAM inference)
//                 0: asynchronous / combinational (LUT ROM)
//
// IMPORTANT: when REGISTERED=1 the output is valid ONE clock cycle after addr.
// bf16_linear_approx accounts for this by delaying frac_part identically,
// so the multiply always combines data from the same input cycle.
// =============================================================================

module bf16_coeff_rom #(
    parameter int DATA_W    = 42,
    parameter int DEPTH     = 128,
    parameter int ADDR_W    = 7,
    parameter bit REGISTERED = 1'b1
)(
    input  logic              clk,
    input  logic [ADDR_W-1:0] addr,
    output logic [DATA_W-1:0] data
);
    (* rom_style = "block" *) logic [DATA_W-1:0] rom [0:DEPTH-1];

    // -------------------------------------------------------------------------
    // Coefficient table
    // Packed format: {b[20:0], a[20:0]} -- both 1.20 unsigned
    // Generated from bf16_exp2_packed_coeffs.hpp
    // -------------------------------------------------------------------------
    initial begin
        rom[  0] = 42'h1b22a659157;
        rom[  1] = 42'h1b31fa59915;
        rom[  2] = 42'h1b41445a0dd;
        rom[  3] = 42'h1b50865a8b0;
        rom[  4] = 42'h1b5fba5b08d;
        rom[  5] = 42'h1b6ee65b876;
        rom[  6] = 42'h1b7e065c06a;
        rom[  7] = 42'h1b8d1c5c868;
        rom[  8] = 42'h1b9c265d072;
        rom[  9] = 42'h1bab265d887;
        rom[ 10] = 42'h1bba185e0a8;
        rom[ 11] = 42'h1bc9005e8d3;
        rom[ 12] = 42'h1bd7dc5f10a;
        rom[ 13] = 42'h1be6aa5f94c;
        rom[ 14] = 42'h1bf56c6019a;
        rom[ 15] = 42'h1c0422609f4;
        rom[ 16] = 42'h1c12ca61258;
        rom[ 17] = 42'h1c216661ac9;
        rom[ 18] = 42'h1c2ff262345;
        rom[ 19] = 42'h1c3e7262bce;
        rom[ 20] = 42'h1c4ce463462;
        rom[ 21] = 42'h1c5b4663d02;
        rom[ 22] = 42'h1c699a645ae;
        rom[ 23] = 42'h1c77e064e66;
        rom[ 24] = 42'h1c86166572a;
        rom[ 25] = 42'h1c943c65ffa;
        rom[ 26] = 42'h1ca252668d6;
        rom[ 27] = 42'h1cb058671bf;
        rom[ 28] = 42'h1cbe4e67ab5;
        rom[ 29] = 42'h1ccc34683b6;
        rom[ 30] = 42'h1cda0a68cc4;
        rom[ 31] = 42'h1ce7cc695df;
        rom[ 32] = 42'h1cf57e69f07;
        rom[ 33] = 42'h1d031e6a83b;
        rom[ 34] = 42'h1d10ae6b17c;
        rom[ 35] = 42'h1d1e286baca;
        rom[ 36] = 42'h1d2b926c424;
        rom[ 37] = 42'h1d38e86cd8c;
        rom[ 38] = 42'h1d462c6d701;
        rom[ 39] = 42'h1d535a6e083;
        rom[ 40] = 42'h1d60766ea12;
        rom[ 41] = 42'h1d6d7c6f3af;
        rom[ 42] = 42'h1d7a706fd59;
        rom[ 43] = 42'h1d874e70710;
        rom[ 44] = 42'h1d9416710d5;
        rom[ 45] = 42'h1da0c871aa7;
        rom[ 46] = 42'h1dad6672487;
        rom[ 47] = 42'h1db9ec72e75;
        rom[ 48] = 42'h1dc65e73870;
        rom[ 49] = 42'h1dd2b67427a;
        rom[ 50] = 42'h1ddefa74c91;
        rom[ 51] = 42'h1deb24756b7;
        rom[ 52] = 42'h1df736760ea;
        rom[ 53] = 42'h1e033276b2c;
        rom[ 54] = 42'h1e0f147757c;
        rom[ 55] = 42'h1e1adc77fda;
        rom[ 56] = 42'h1e268c78a47;
        rom[ 57] = 42'h1e3222794c2;
        rom[ 58] = 42'h1e3da079f4c;
        rom[ 59] = 42'h1e49007a9e4;
        rom[ 60] = 42'h1e54487b48b;
        rom[ 61] = 42'h1e5f747bf41;
        rom[ 62] = 42'h1e6a847ca06;
        rom[ 63] = 42'h1e75787d4da;
        rom[ 64] = 42'h1e7ef67de60;
        rom[ 65] = 42'h1e89b67e950;
        rom[ 66] = 42'h1e94587f44f;
        rom[ 67] = 42'h1e9edc7ff5e;
        rom[ 68] = 42'h1ea94280a7c;
        rom[ 69] = 42'h1eb38a815a9;
        rom[ 70] = 42'h1ebdb4820e6;
        rom[ 71] = 42'h1ec7be82c33;
        rom[ 72] = 42'h1ed1aa8378f;
        rom[ 73] = 42'h1edb76842fb;
        rom[ 74] = 42'h1ee52084e77;
        rom[ 75] = 42'h1eeeaa85a03;
        rom[ 76] = 42'h1ef8128659f;
        rom[ 77] = 42'h1f015a8714b;
        rom[ 78] = 42'h1f0a8087d07;
        rom[ 79] = 42'h1f1382888d4;
        rom[ 80] = 42'h1f1c62894b1;
        rom[ 81] = 42'h1f251e8a09e;
        rom[ 82] = 42'h1f2db68ac9c;
        rom[ 83] = 42'h1f362c8b8ab;
        rom[ 84] = 42'h1f3e7a8c4cb;
        rom[ 85] = 42'h1f45a28cf74;
        rom[ 86] = 42'h1f4cac8da2b;
        rom[ 87] = 42'h1f54948e679;
        rom[ 88] = 42'h1f5c568f2d8;
        rom[ 89] = 42'h1f63f08ff48;
        rom[ 90] = 42'h1f6b6490bca;
        rom[ 91] = 42'h1f72b09185d;
        rom[ 92] = 42'h1f79d292502;
        rom[ 93] = 42'h1f80cc931b8;
        rom[ 94] = 42'h1f879c93e80;
        rom[ 95] = 42'h1f8e4494b59;
        rom[ 96] = 42'h1f952695914;
        rom[ 97] = 42'h1f9b7496612;
        rom[ 98] = 42'h1fa19897323;
        rom[ 99] = 42'h1fa78e98046;
        rom[100] = 42'h1fad5a98d7b;
        rom[101] = 42'h1fb2f699ac2;
        rom[102] = 42'h1fb8669a81c;
        rom[103] = 42'h1fbda89b588;
        rom[104] = 42'h1fc2ba9c307;
        rom[105] = 42'h1fc79e9d099;
        rom[106] = 42'h1fcc089dd63;
        rom[107] = 42'h1fd0489ea3d;
        rom[108] = 42'h1fd4a29f806;
        rom[109] = 42'h1fd8c8a05e1;
        rom[110] = 42'h1fdcbea13d0;
        rom[111] = 42'h1fe082a21d2;
        rom[112] = 42'h1fe43aa3090;
        rom[113] = 42'h1fe788a3e82;
        rom[114] = 42'h1feaaea4cc0;
        rom[115] = 42'h1feda0a5b11;
        rom[116] = 42'h1ff05ca6976;
        rom[117] = 42'h1ff2cea777b;
        rom[118] = 42'h1ff50ea8593;
        rom[119] = 42'h1ff728a9433;
        rom[120] = 42'h1ff91eaa37a;
        rom[121] = 42'h1ffabcab1ec;
        rom[122] = 42'h1ffc28ac08e;
        rom[123] = 42'h1ffd5cacf44;
        rom[124] = 42'h1ffe62aded1;
        rom[125] = 42'h1fff1eaed67;
        rom[126] = 42'h1fffa8afce8;
        rom[127] = 42'h1ffff2b0c81;
    end

    // -------------------------------------------------------------------------
    // Read port
    //   REGISTERED=1 : synchronous output  -> BRAM inference (1-cycle latency)
    //   REGISTERED=0 : combinational output -> LUT ROM        (0-cycle latency)
    // -------------------------------------------------------------------------
    generate
        if (REGISTERED) begin : gen_sync
            always_ff @(posedge clk)
                data <= rom[addr];
        end else begin : gen_async
            assign data = rom[addr];
        end
    endgenerate

endmodule : bf16_coeff_rom
