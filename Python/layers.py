import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryConv2dFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
		w = torch.sign(weight)
		output = F.conv2d(input, w,  bias, stride, padding, dilation, groups)
		ctx.save_for_backward(input, w, bias) 
		ctx.stride = stride
		ctx.padding = padding
		ctx.dilation = dilation
		ctx.groups = groups

		return output

	@staticmethod
	def backward(ctx, grad_output):

		input, w, bias = ctx.saved_tensors 
		stride = ctx.stride
		padding = ctx.padding
		dilation = ctx.dilation
		groups = ctx.groups
		grad_input = grad_weight = grad_bias = None

		if ctx.needs_input_grad[0]:
				grad_input = torch.nn.grad.conv2d_input(input.shape, w, grad_output, stride, padding, dilation, groups) 
		if ctx.needs_input_grad[1]:
				grad_weight = torch.nn.grad.conv2d_weight(input, w.shape, grad_output, stride, padding, dilation, groups) 
		if bias is not None and ctx.needs_input_grad[2]:
				grad_bias = grad_output.sum((0,2,3)).squeeze(0)

		return grad_input, grad_weight, grad_bias, None, None, None, None

class BinaryConv2dLayer(nn.Module):
  def __init__(self, out_channel, in_channel, filter_size):
    super().__init__()

    weight = torch.empty(out_channel * in_channel * filter_size * filter_size, requires_grad = True).reshape(out_channel, in_channel, filter_size, filter_size)
    torch.nn.init.normal_(weight, mean=0, std=1)
    self.weight = nn.Parameter(weight)

    self.fn = BinaryConv2dFunction.apply

  def forward(self, x):
    return self.fn(x, self.weight)

class StepActivation(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return torch.where(input >= 0, 1., -1.)
    #return torch.sign(input)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    #grad_input = torch.where(-1 < input and input < 1, input, torch.zeros(len(input), len(input[0])))
    grad_input = 1 / torch.cosh(input ** 2) * grad_output.clone()
    return grad_input

class BinaryLinearFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, weight, bias):
    w = torch.sign(weight)
    b = torch.sign(bias)
    ctx.save_for_backward(input, w, b)
    return torch.mm(input, torch.t(w)) + b.unsqueeze(0).expand(input.shape[0], w.shape[0])

  @staticmethod
  def backward(ctx, grad_output):
    input, w, b = ctx.saved_tensors
    grad_input = torch.mm(grad_output, w.clone())
    grad_weight = torch.mm(torch.t(grad_output), input.clone())
    grad_bias = grad_output.sum(0)
    return grad_input, grad_weight, grad_bias

class BinaryLinearLayer(nn.Module):
  def __init__(self, input, output):
    super().__init__()

    weight = torch.empty(output, input, requires_grad = True)
    nn.init.normal_(weight, mean=0, std=1)
    self.weight = nn.Parameter(weight)

    bias = torch.empty(output, requires_grad = True)
    nn.init.normal_(bias, mean = 0, std = 1)
    self.bias = nn.Parameter(bias)

    self.fn = BinaryLinearFunction.apply

  def forward(self, x):
    return self.fn(x, self.weight, self.bias)
