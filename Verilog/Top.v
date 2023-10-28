module top();
	
	reg CLK, NRST, NEXT;

	reg next;
	reg state;

	always @(posedge CLK) begin
		if(!NEXT) begin
			if(state == 1) begin
				next <= 0;
			end else begin
				next <= 1;
				state <= 1;
			end
		end else begin
			next <= 0;
			state <= 0;
		end 
	end


	wire [783:0] input_node;
	wire input_finish;
	wire [576:0] conv_node;
	wire conv_finish;
	wire [128:0] hidden_node;
	wire hidden_finish;
	wire [3:0] output_result;
	wire output_finish;

	always #10 CLK <= ~CLK;

	initial begin
		CLK <= 0;
		NRST <= 0;
		NEXT <= 1;
		#100 NRST <= 1;
		#30000 NEXT <= 0;
		#100 NEXT <= 1;
		#30000 NEXT <= 0;
		#30000 $finish;
	end

	initial begin
		$monitor("%b,  %b, %b", InputLayer.image_addr, input_node, next);
		$monitor("time : %t, result : %d output_finish : %d", $time, output_result, output_finish); 
//	$monitor("hidden_node : %b", hidden_node); 		
	end

	input_layer InputLayer(.CLK(CLK), .NRST(NRST), .next(next), .input_node(input_node), .input_finish(input_finish));
	conv_layer ConvLayer(.CLK(CLK), .NRST(NRST), .next(next), .input_node(input_node), .input_finish(input_finish), .conv_node(conv_node), .conv_finish(conv_finish));
	hidden_layer HiddenLayer(.CLK(CLK), .NRST(NRST), .next(next), .conv_node(conv_node), .conv_finish(conv_finish), .hidden_node(hidden_node), .hidden_finish(hidden_finish));
	output_layer OutputLayer(.CLK(CLK), .NRST(NRST), .next(next), .hidden_node(hidden_node), .hidden_finish(hidden_finish), .output_result(output_result), .output_finish(output_finish));
	

endmodule
