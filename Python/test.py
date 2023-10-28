import model
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import os

model = model.BinaryCNN();

# load
path = "model_weight.pth"
if os.path.isfile(path):	
	model.load_state_dict(torch.load(path)) 

# MNIST datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(len(train_images), 28*28).astype("float32") / 255
test_images = test_images.reshape(len(test_images), 28*28).astype("float32") / 255
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Binalize images and Create dataloader
train_x = torch.tensor(train_images, dtype=torch.float32).reshape(-1, 1, 28, 28)
train_x = torch.where(train_x > 0.2, 1., -1.)
train_t = torch.tensor(train_labels, dtype=torch.float32)
train_dataset = Data.TensorDataset(train_x, train_t)
test_x = torch.tensor(test_images, dtype=torch.float32).reshape(-1, 1, 28, 28)
test_x = torch.where(test_x > 0.2, 1., -1.)
test_t = torch.tensor(test_labels, dtype=torch.float32)
test_dataset = Data.TensorDataset(test_x, test_t)
batch_size = 1000
train_dataloader = Data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = Data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# test
def test_step(x, t):
	model.eval()
	preds = model(x)
	return preds

count = 0
correct = 0
for (x, t) in test_dataloader:
	results = model(x).detach().numpy()
	for i in range(len(results)):
		result = np.argmax(results[i])
		ref = np.argmax(t[i].detach().numpy())
		if result == ref:
			correct += 1
		count += 1
print("correct rate : ", correct / count)

