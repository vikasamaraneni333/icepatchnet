import torch
import torch.nn as nn

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.ReLU1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.ReLU2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.ReLU3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.conv2(x)
        x = self.ReLU2(x)
        x = self.conv3(x)
        output = self.ReLU3(x)
        return output
