import torch
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class StateCNN(nn.Module):
    def __init__(self):
        super(StateCNN, self).__init__()

        self.pool = nn.MaxPool2d(5, 5)      # Max Pooling
        self.dropConv = nn.Dropout2d(0.3)    # Dropout for conv layers
        self.dropFC = nn.Dropout(0.3)       # Dropout for FC layers

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 6 * 6, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 20)

    def forward(self, x):
        # Conv layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropConv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropConv(x)

        x = x.view(-1, 32 * 6 * 6)

        # FC layers
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropFC(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropFC(x)
        out_put = self.out(x)

        return out_put


