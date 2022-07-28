"""
The deep learning model to predict wrist movements
Dataset limited to shape (373, 9). Where 373 is the total number of frames.
9 is the size of feature vector. (Root, Thumb 0, Pinky 0). Each sub-vector is the x, y, z coordinate.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from csv_utils import read_csv
from pathlib import Path


class MovementDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        self.coordinates = []
        self.labels = []
        for csv_file in Path(file_path).iterdir():
            coordinates, movements = read_csv(csv_file)
            extracted_frame = coordinates[:, 3:6]  # rootPos
            extracted_frame = np.hstack(
                (extracted_frame, coordinates[:, 12:15])
            )  # Thumb 0
            extracted_frame = np.hstack(
                (extracted_frame, coordinates[:, 51:54])
            )  # Pinky 0
            self.coordinates.append(extracted_frame)
            self.labels.append(movements[1])
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


# Fully connected network
class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(3357, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 8)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Convolutional neural network
class ConvolutionNet(nn.Module):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(32 * 373 * 9, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    train_loader = DataLoader(
        MovementDataset("../train_data", transform=transformation),
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        MovementDataset("../test_data", transform=transformation),
        batch_size=64,
    )

    # training details
    epochs = 150
    lr = 1e-3
    betas = (0.9, 0.999)
    loss_func = nn.CrossEntropyLoss()
    model = FullyConnectedNet()
    # model = ConvolutionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

    # training
    for epoch in range(epochs):  # loop over the dataset multiple times
        # set the model to training mode
        model.train()
        # cumulative data for record the loss and accuracy changes as training proceeds
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # loop over mini-batches in the dataset
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # recording the training accuracy and loss for the mini-batch
            train_total += labels.size(0)
            train_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            running_loss += loss.item()

        train_accuracy = train_correct / train_total

        # Validation
        validation_loss = 0.0
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()
            for i, (inputs, labels) in enumerate(val_loader, 0):
                # forward + loss calculation
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

                # recording the validation accuracy and loss for the mini-batch
                val_total += labels.size(0)
                val_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                validation_loss += loss.item()

            # calculate the validation loss and accuracy
            validation_loss /= len(val_loader)
            validation_accuracy = val_correct / val_total
            print(f"Validation accuracy: {validation_accuracy}")

    # torch.save(model.state_dict(), "../models/conv.pt")
    # torch.save(model.state_dict(), "../models/fc.pt")
