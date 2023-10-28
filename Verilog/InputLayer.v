
// 	draw  u8 (.CLK(CLK40), .NRST(NRST), .X(posx), .Y(posy), .R(posr), .G(posg), .B(posb), .IMAGE_OUT(), .IMAGE_ADDR(image_addr), .imout(image_out));

module input_layer(CLK, NRST, next, input_node, input_finish);
	input CLK, NRST;
	input next;
	output [783:0] input_node;
	output input_finish;

	reg [9:0] image_addr;
	reg input_finish;

	always @(posedge CLK) begin
		if(!NRST) begin
			image_addr <= 0;
			input_finish <= 0;
		end else if(next) begin
			image_addr <= image_addr + 1;
			input_finish <= 0;
		end else if(input_finish == 0) begin
			input_finish <= 1;
		end
	end


	
	// 1 image : 28*28 pixel (784 bit)
	// bus width : 8 bit
	// address : 0 ~ 111 (7 bit)
	// size : 8 bit * 128
	image_rom ImageRom(.address(image_addr), .clock(CLK), .q(input_node));
	 
endmodule

//this module is used in simulation
module image_rom (clock, address, q);
	input clock;
	input [9:0] address;
	output [783:0] q;

	reg [783:0] memory [0:1023];

	assign q = memory[address];

	initial begin
		$readmemb("test_image.dat", memory);
	end
endmodule
