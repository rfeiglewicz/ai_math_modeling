// =============================================================================
// bf16_pipe_pad.sv
// Plain delay line used to pad a core's output up to the common pipeline
// depth (bf16_exp2_pkg::UNIFIED_PIPE_DEPTH).
//
// The cores have different natural latencies. Padding the shallow ones keeps
// every implementation cycle-for-cycle identical, so one can be swapped for
// another without changing anything downstream.
//
// STAGES = 0 collapses to a wire, so the default costs nothing.
//
// RESET_DATAPATH = 0 drops the asynchronous reset. That is safe because the
// data is only ever consumed when the accompanying valid bit is high, and the
// valid shift register in the parent core IS reset. Without the reset Vivado
// is free to implement the chain as an SRL16E instead of discrete flip-flops.
// =============================================================================

module bf16_pipe_pad #(
    parameter int STAGES         = 0,
    parameter int WIDTH          = 16,
    parameter bit RESET_DATAPATH = 1'b1
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             pipe_en,
    input  logic [WIDTH-1:0] d,
    output logic [WIDTH-1:0] q
);

    generate
        if (STAGES <= 0) begin : gen_bypass
            assign q = d;
        end else begin : gen_delay
            logic [WIDTH-1:0] chain [STAGES];

            always_comb q = chain[STAGES-1];

            for (genvar i = 0; i < STAGES; i++) begin : gen_stage
                logic [WIDTH-1:0] nxt;
                assign nxt = (i == 0) ? d : chain[i-1];

                if (RESET_DATAPATH) begin : gen_reg_rst
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n)       chain[i] <= '0;
                        else if (pipe_en) chain[i] <= nxt;
                    end
                end else begin : gen_reg_nrst
                    always_ff @(posedge clk) begin
                        if (pipe_en) chain[i] <= nxt;
                    end
                end
            end
        end
    endgenerate

endmodule : bf16_pipe_pad
