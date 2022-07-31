"""
The deep learning model to predict wrist movements
Dataset limited to shape (359, 9). Where 373 is the total number of frames.
9 is the size of feature vector. (Root, Thumb 0, Pinky 0). Each sub-vector is the x, y, z coordinate.
Written by: Xingjian Leng on 28, Jul, 2022
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
            # TODO: Improvement
            extracted_frame = coordinates[:, 3:6]  # rootPos
            extracted_frame = np.hstack(
                (extracted_frame, coordinates[:, 12:15])
            )  # Thumb 0
            extracted_frame = np.hstack(
                (extracted_frame, coordinates[:, 51:54])
            )  # Pinky 0
            for index in (5, 8, 11, 14, 18):
                extracted_frame = np.hstack(
                    (extracted_frame, coordinates[:, index : index + 3])
                )
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
        self.fc1 = nn.Linear(359 * 24, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 8)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Convolutional neural network
class ConvolutionNetMovement(nn.Module):
    def __init__(self):
        super(ConvolutionNetMovement, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.fc1 = nn.Linear(8 * 359 * 24, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    mode = "CNN"
    assert mode in {"CNN", "FC"}

    transformation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    val_loader = DataLoader(
        MovementDataset("../validation_data", transform=transformation),
        batch_size=64,
    )

    # neural network model
    model = ConvolutionNetMovement() if mode == "CNN" else FullyConnectedNet()

    if not (
        Path("../models/conv_movement.pt").exists()
        and mode == "CNN"
        or Path("../models/fc_movement.pt").exists()
        and mode == "FC"
    ):
        # when there is no existing model state saved, train a new model
        train_loader = DataLoader(
            MovementDataset("../train_data", transform=transformation),
            batch_size=64,
            shuffle=True,
        )

        # training details
        epochs = 60
        lr = 1e-3
        betas = (0.9, 0.999)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        best_val_acc = 0.0
        best_dict = None

        # training
        for epoch in range(epochs):  # loop over the dataset multiple times
            # set the model to training mode
            model.train()
            # cumulative data for record the loss and accuracy changes as training proceeds
            running_loss, train_correct, train_total = 0.0, 0, 0

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
            print(f"\nEpoch: {epoch + 1}\n----------\nTraining loss: {running_loss}")
            print(f"Training accuracy: {train_accuracy}")

            # Validation
            validation_loss, val_total, val_correct = 0.0, 0, 0

            with torch.no_grad():
                # set the model to evaluation mode
                model.eval()
                for i, (inputs, labels) in enumerate(val_loader, 0):
                    # forward + loss calculation
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                    # recording the validation accuracy and loss for the mini-batch
                    val_total += labels.size(0)
                    val_correct += (
                        (torch.max(outputs.data, 1)[1] == labels).sum().item()
                    )
                    validation_loss += loss.item()

                # calculate the validation loss and accuracy
                validation_loss /= len(val_loader)
                validation_accuracy = val_correct / val_total
                if validation_accuracy > best_val_acc:
                    best_dict = model.state_dict()
                    best_val_acc = validation_accuracy
                print(f"Validation loss: {validation_loss}")
                print(f"Validation accuracy: {validation_accuracy}\n----------\n")

        if mode == "CNN":
            torch.save(best_dict, "../models/conv_movement.pt")
        else:
            torch.save(best_dict, "../models/fc_movement.pt")
    else:
        # if there are existing models, load it and evaluate it with validation dataset
        if mode == "CNN":
            model.load_state_dict(torch.load("../models/conv_movement.pt"))
        else:
            model.load_state_dict(torch.load("../models/fc_movement.pt"))
        model.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader, 0):
                # forward + loss calculation
                outputs = model(inputs)
                # recording the validation accuracy for the mini-batch
                val_total += labels.size(0)
                val_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            # calculate the validation accuracy
            validation_accuracy = val_correct / val_total
            print(
                f"\n----------\nValidation accuracy: {validation_accuracy}\n----------\n"
            )
