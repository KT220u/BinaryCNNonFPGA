import model
from tensorflow.keras.datasets import mnist
import torch
import os

model = model.BinaryCNN()

# load
path = "model_weight.pth"
if os.path.isfile(path):
	model.load_state_dict(torch.load(path)) 

# weights
conv_weight = model.conv1.weight
conv_weight = torch.where(conv_weight >= 0, 1, 0)
weight1 = model.fc1.weight
weight1 = torch.where(weight1 >= 0, 1, 0)
bias1 = model.fc1.bias
bias1 = torch.where(bias1 >= 0, 1, 0)
weight2 = model.fc2.weight
weight2 = torch.where(weight2 >= 0, 1, 0)
bias2 = model.fc2.bias
bias2 = torch.where(bias2 >= 0, 1, 0)

# image
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape(len(test_images), 28*28).astype("float32") / 255
test_x = torch.tensor(test_images, dtype=torch.float32)
test_x = torch.where(test_x > 0.2, 1, 0)

conv_node = 576 # 24*24 
filter_size = 5
hidden_node = 128
output_node = 10

def Weight1ToDat(weight, bias, f):
	for i in range(conv_node):
		for k in range(hidden_node):
			f.write(str(int(weight[k][i])))
		f.write("\n")

	for k in range(hidden_node):
		f.write(str(int(bias[k])))
	f.write("\n")

def Weight2ToDat(weight, bias, f):
	for i in range(hidden_node):
		for j in range(output_node):
			f.write(str(int(weight[j][i])))
		f.write("\n")
	for j in range(output_node):
		f.write(str(int(bias[j])))
	f.write("\n")

def ImageToDat(image, f):
	for i in range(len(image)):
		num = image[len(image) - 1 - i]
		f.write(str(int(num)))
	f.write("\n")

def ConvWeightToDat(weight, f):
	for i in range(filter_size):
		for j in range(filter_size):
			f.write(str(int(weight[0][0][i][4-j])))
		f.write("\n")

with open("../Verilog/test_image.dat", 'w') as f:
  ImageToDat(test_x[0], f)
with open("../Verilog/test_image.dat", 'a') as f:
	for i in range(1, 1024):
		ImageToDat(test_x[i], f)
	
with open("../Verilog/hidden_weight.dat", "w") as f:
  Weight1ToDat(weight1, bias1, f)

with open("../Verilog/output_weight.dat", "w") as f:
  Weight2ToDat(weight2, bias2, f)

with open("../Verilog/conv_weight.dat", "w") as f:
  ConvWeightToDat(conv_weight, f)

