import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

# hyperparams
config = dict(
    model_path="model.pt",
    lr=1e-3,
    weight_decay=1e-6,
    n_epochs=8,
    num_workers=0,
    batch_size=64,
    train_test_ratio=0.8,
    seed=1234,
)

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_transform():
    return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(640, 400)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.dropout(x)
        x = x.reshape(batch_size, -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def train_fn(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for batch in train_loader:
        # load data to device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)

        # feed forwards
        output = model(x)

        # compute loss
        loss = criterion(output, y)
        train_loss += loss.item() / batch_size

        # backprop and step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return train_loss


def eval_fn(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    n_correct = 0
    n_total = 0
    for batch in val_loader:
        # load data to device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)

        with torch.no_grad():
            # feed forward
            output = model(x)

            # compute loss
            loss = criterion(output, y)
            val_loss += loss.item() / batch_size

            # get predictions
            pred = torch.argmax(output, axis=-1)

            # add number of correct and total predictions made
            n_correct += torch.sum(pred == y).item()
            n_total += batch_size

    val_acc = n_correct / n_total
    return val_loss, val_acc


def train():
    # load train dataset
    transform = get_transform()
    train_dataset = MNIST(
        root="data/mnist/train", train=True, download=True, transform=transform
    )

    # train/validation split
    train_size = int(len(train_dataset) * config["train_test_ratio"])
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # model
    model = CNN()
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # train loop
    for epoch in range(config["n_epochs"]):
        # train
        train_loss = train_fn(model, train_loader, criterion, optimizer)

        # eval
        val_loss, val_acc = eval_fn(model, val_loader, criterion)

        # log results
        print(
            f"Epoch {epoch+1}/{config['n_epochs']}\t"
            f"loss {train_loss:.3f}\t"
            f"val_loss {val_loss:.3f}\t"
            f"val_acc {val_acc:.3f}\t"
        )

    # save model
    torch.save(model.state_dict(), config["model_path"])


def evaluate():
    # load test dataset
    transform = get_transform()
    test_dataset = MNIST(
        root="data/mnist/test", train=False, download=True, transform=transform
    )

    # data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # load model
    model = CNN()
    model.load_state_dict(torch.load(config["model_path"]))
    model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # test
    test_loss, test_acc = eval_fn(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test Loss: {test_loss:.3f}")


if __name__ == "__main__":
    set_seed(config["seed"])
    train()
    evaluate()
