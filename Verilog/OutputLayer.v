module output_layer(CLK, NRST, next, hidden_node, hidden_finish, output_result, output_finish);
	input CLK, NRST, next;
	input [128:0] hidden_node;
	input hidden_finish;
	output [3:0] output_result;
	output output_finish;

	reg output_finish;
	reg [7:0] hidden_index;
	wire [9:0] weight_buffer;
	reg state;
	wire [7:0] output_value [0:9];

	wire [3:0] result1, result2, result3, result4, result5;

	assign result1 = ((output_value[0] > output_value[1]) & (output_value[0] > output_value[2])) ? 0 : (output_value[1] > output_value[2]) ? 1 : 2;
	assign result2 = ((output_value[3] > output_value[4]) & (output_value[3] > output_value[5])) ? 3 : (output_value[4] > output_value[5]) ? 4 : 5;
	assign result3 = ((output_value[6] > output_value[7]) & (output_value[6] > output_value[8])) ? 6 : (output_value[7] > output_value[2]) ? 7 : 8;
	assign result4 = (output_value[result1] > output_value[result2]) ? result1 : result2;
	assign result5 = (output_value[result3] > output_value[9]) ? result3 : 9;
	assign output_result = (output_value[result4] > output_value[result5]) ? result4 : result5;
	
	

	always @(posedge CLK) begin
		if(!NRST | next) begin
			output_finish <= 0;
			hidden_index <= 8'b11111111;
			state <= 0;
		end else if(hidden_finish & ~output_finish) begin
			if(state == 0) begin
				state <= 1;
				hidden_index <= hidden_index + 1;
			end else if(state == 1) begin
				if(hidden_index == 128) begin
					output_finish <= 1;
				end 
				state <= 0;
			end
		end
	end


	generate
		genvar i;
		for(i = 0; i <= 9; i = i + 1) begin : Block1
			output_node OutputNode(.CLK(CLK), .NRST(NRST), .next(next), .state(state), .output_in1(hidden_node[hidden_index]), .output_in2(weight_buffer[9-i]), .output_value(output_value[i]));
		end
	endgenerate


	weight2_rom Weight2Rom(.clock(CLK), .address(hidden_index), .q(weight_buffer));

endmodule

module output_node(CLK, NRST, next, state, output_in1, output_in2, output_value);
	input CLK, NRST, next, state;
	input output_in1, output_in2;
	output [7:0] output_value;
	
	reg [7:0] output_value;
	
	always @(posedge CLK) begin
		if(!NRST | next) begin
			output_value <= 0;
		end else if(state) begin
			output_value <= output_value + {7'b0, ~(output_in1 ^ output_in2)};
		end
	end
endmodule

module weight2_rom(clock, address, q);
	input clock;
	input [7:0] address;
	output [9:0] q;

	reg [9:0] memory [0:128];
	assign q = memory[address];

	initial begin
		$readmemb("output_weight.dat", memory);
	end
endmodule

	
