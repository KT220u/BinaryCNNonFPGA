import layers
import torch.nn as nn

class BinaryCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = layers.BinaryConv2dLayer(1, 1, 5);
    self.step1 = layers.StepActivation.apply
    self.fc1 = layers.BinaryLinearLayer(24*24, 128)
    self.step2 = layers.StepActivation.apply
    self.fc2 = layers.BinaryLinearLayer(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.step1(x)
    x = x.view(-1, 24*24)
    x = self.fc1(x)
    x = self.step2(x)
    x = self.fc2(x)
    return x
