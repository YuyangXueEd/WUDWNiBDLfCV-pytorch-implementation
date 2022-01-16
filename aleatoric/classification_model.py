import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="Aleatoric example")
parser.add_argument('--cuda', action='store_true', help='Use GPU')
parser.add_argument('--data', type=str, default='./data/', help='the path of the dataset')
parser.add_argument('--batch_size', type=int, default=128, help="Number per batch")
parser.add_argument('--classes', type=int, default=10, help="how many classes for the task")
args = parser.parse_args()

DOWNLOAD_MNIST = False

if not os.path.exists(args.data) or not os.listdir(args.data):
    DOWNLOAD_MNIST = True

train_loader = data.DataLoader(
    datasets.MNIST(args.data,
                   train=True,
                   download=DOWNLOAD_MNIST,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size,
    shuffle=True
)

test_loader = data.DataLoader(
    datasets.MNIST(args.data,
                   train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size,
    shuffle=False
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input (1, 28, 28)
                out_channels=16,
                kernel_size=5,
                stride=1,
                # if you want same width and length of this image after Conv2d,
                # padding=(kernel_size-1)/2 if stride=1
                padding=2
            ),  # output (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output (16, 14, 14)
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input (16, 14, 14)
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # output (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output (32, 7, 7)
            nn.Dropout(0.5)
        )

        self.linear = nn.Linear(32 * 7 * 7, args.classes * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (args.batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        logit = self.linear(x)
        mu, sigma = logit.split(args.classes, 1)
        return mu, sigma


cnn = SimpleCNN()

print(cnn)
