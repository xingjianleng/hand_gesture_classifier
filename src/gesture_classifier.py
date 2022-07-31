"""
For each hand, we can use a 359x5 matrix to encode the finger state.
373 is the total number of frames, 5 (0 - 4) represents each finger
For each element, 0 means straight; 1 means half-curve; 2 means curve.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from csv_utils import read_csv
from finger_classifier import finger_states_encoding
from pathlib import Path


class GestureDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        self.coordinates = []
        self.labels = []
        for csv_file in Path(file_path).rglob("*.csv"):
            coordinates, movements = read_csv(csv_file)
            extracted_data = finger_states_encoding(coordinates)
            self.coordinates.append(extracted_data)
            self.labels.append(movements[0])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        coordinate = self.coordinates[idx]
        label = self.labels[idx]
        if self.transform:
            coordinate = self.transform(coordinate)
        if self.target_transform:
            label = self.target_transform(label)
        return coordinate, label


class FullyConnectedNetGesture(nn.Module):
    def __init__(self):
        super(FullyConnectedNetGesture, self).__init__()
        self.fc1 = nn.Linear(1865, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 8)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConvolutionNetGesture(nn.Module):
    def __init__(self):
        super(ConvolutionNetGesture, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.fc1 = nn.Linear(8 * 373 * 5, 1024)
        self.fc2 = nn.Linear(1024, 15)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
