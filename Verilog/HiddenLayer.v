// HiddenLayer
module hidden_layer(CLK, NRST, next, conv_node, conv_finish, hidden_node, hidden_finish);
	input CLK, NRST, next;
	input [576:0] conv_node;
	input conv_finish;
	output [128:0] hidden_node;
	output [9:0] hidden_value_out;
	output hidden_finish;
	
	reg [9:0] conv_index;
	reg state;
	reg hidden_finish;

	wire [127:0] weight_buffer;

	assign hidden_node[128] = 1;

	always @(posedge CLK) begin
		if(!NRST | next) begin
			conv_index <= 10'b1111111111;
			hidden_finish <= 0;
			state <= 0;
		end else if(conv_finish & ~hidden_finish) begin
			if(state == 0) begin
				state <= 1;
				conv_index <= conv_index + 1;
			end else if(state == 1) begin
				if(conv_index == 576) begin
					hidden_finish <= 1;
				end 
				state <= 0;
			end
		end
	end

	generate
		genvar i;
		for(i = 0; i <= 127; i = i + 1) begin : Block1
			hidden_node HiddenNode(.CLK(CLK), .NRST(NRST), .next(next), .state(state), .hidden_in1(weight_buffer[127-i]), .hidden_in2(conv_node[conv_index]), .hidden_out(hidden_node[i]));
		end
	endgenerate

	weight1_rom Weight1Rom(.clock(CLK), .address(conv_index), .q(weight_buffer));
endmodule

module hidden_node(CLK, NRST, next, state, hidden_in1, hidden_in2, hidden_out);
	input CLK, NRST, next, state;
	input hidden_in1, hidden_in2;
	output hidden_out;
	
	reg [9:0] hidden_value;

	assign hidden_out = (hidden_value >= 294) ? 1 : 0;
	
	always @(posedge CLK) begin
		if(!NRST | next) begin
			hidden_value <= 0;
		end else if(state) begin
			hidden_value <= hidden_value + {9'b0, ~(hidden_in1 ^ hidden_in2)};
		end
	end
endmodule


// 128 * 784 weights and 128 bias
// bus width : 128 bit
// address : 0 ~ 784 ( 10 bit )
module weight1_rom(clock, address, q);
	input clock;
	input [9:0] address;
	output [127:0] q;

	reg [127:0] memory [0:576];
	assign q = memory[address];

	initial begin
		$readmemb("hidden_weight.dat", memory);
	end
endmodule
