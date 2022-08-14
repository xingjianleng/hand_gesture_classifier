"""
For each hand, we can use a 359x5 matrix to encode the finger state.
373 is the total number of frames, 5 (0 - 4) represents each finger
For each element, 0 means straight; 1 means half-curve; 2 means curve.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from copy import deepcopy
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
        self.fc1 = nn.Linear(359 * 5, 512)
        self.fc2 = nn.Linear(512, 15)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ConvolutionNetGesture(nn.Module):
    def __init__(self):
        super(ConvolutionNetGesture, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.fc1 = nn.Linear(4 * 359 * 5, 512)
        self.fc2 = nn.Linear(512, 15)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
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
        GestureDataset("../validation_data", transform=transformation),
        batch_size=64,
    )

    # neural network model
    model = ConvolutionNetGesture() if mode == "CNN" else FullyConnectedNetGesture()

    if not (
        Path("../models/conv_gesture.pt").exists()
        and mode == "CNN"
        or Path("../models/fc_gesture.pt").exists()
        and mode == "FC"
    ):
        # when there is no existing model state saved, train a new model
        train_loader = DataLoader(
            GestureDataset("../train_data", transform=transformation),
            batch_size=64,
            shuffle=True,
        )

        # training details
        epochs = 15
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
                    best_dict = deepcopy(model.state_dict())
                    best_val_acc = validation_accuracy
                print(f"Validation loss: {validation_loss}")
                print(f"Validation accuracy: {validation_accuracy}\n----------\n")

        if mode == "CNN":
            torch.save(best_dict, "../models/conv_gesture.pt")
        else:
            torch.save(best_dict, "../models/fc_gesture.pt")
    else:
        # if there are existing models, load it and evaluate it with validation dataset
        if mode == "CNN":
            model.load_state_dict(torch.load("../models/conv_gesture.pt"))
        else:
            model.load_state_dict(torch.load("../models/fc_gesture.pt"))
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
