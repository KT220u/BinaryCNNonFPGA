module conv_layer(CLK, NRST, next, input_node, input_finish, conv_node, conv_finish);
	input CLK, NRST, next;
	input [783:0] input_node;
	input input_finish;
	output [576:0] conv_node; // 24 * 24 bit + 1 bit (const 1)
	output conv_finish;

	wire [27:0] input_node_square [0:27];
	reg conv_finish;
	reg [4:0] weight [0:4];
	reg [2:0] weight_index_row;
	reg [2:0] weight_index_col;
	reg state;

	wire [4:0] conv_value [0:576];

	assign conv_node[576] = 1;

	generate
		genvar i;
		genvar j;
		for(i = 0; i <= 27; i = i + 1) begin : Block1
			for(j = 0; j <= 27; j = j + 1) begin : Block11
				assign input_node_square[i][j]  = input_node[28 * i + j];
			end
		end
	endgenerate

	generate
		genvar k;
		genvar l;
		for(k = 0; k <= 23; k = k + 1) begin : Block2
			for(l = 0; l <= 23; l = l + 1) begin : Block21
				conv_node ConvNode(.CLK(CLK), .NRST(NRST), .next(next), .state(state),
				.conv_in1(input_node_square[weight_index_row + k][weight_index_col + l]), 
				.conv_in2(weight[weight_index_row][weight_index_col]),
				.conv_out(conv_node[24 * k + l]),
				.conv_value(conv_value[24 * k + l]));
			end
		end
	endgenerate
	
	initial begin
		$readmemb("conv_weight.dat", weight);
	end

	always @(posedge CLK) begin
		if (!NRST |  next) begin
			weight_index_row <= 0;
			weight_index_col <= 3'b111;
			state <= 0;
			conv_finish <= 0;
		end else if (input_finish & ~conv_finish) begin
			if (state == 0) begin
				if (weight_index_col == 4) begin
					weight_index_col <= 0;
					weight_index_row <= weight_index_row + 1;
				end else begin
					weight_index_col <= weight_index_col + 1;
				end
				state <= 1;
			end else if (state == 1) begin
				state <= 0;
				if(weight_index_row == 4 & weight_index_col == 4) begin
					conv_finish <= 1;
				end
			end
		end
	end

endmodule

module conv_node(CLK, NRST, next, state, conv_in1, conv_in2, conv_out, conv_value);
	input CLK, NRST, next, state;
	input conv_in1;
	input conv_in2;
	output conv_out;
	output [4:0] conv_value;

	reg [4:0] conv_value;

	assign conv_out = (conv_value >= 13) ? 1 : 0;

	always @(posedge CLK) begin
		if (!NRST | next) begin
			conv_value <= 0;
		end else if (state == 1) begin
			conv_value <= conv_value + {4'b0, ~(conv_in1 ^ conv_in2)};
		end
	end
endmodule
