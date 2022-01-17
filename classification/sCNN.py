import os
from importlib import import_module

from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, classes, wa):
        super(SimpleCNN, self).__init__()
        self.classes = classes
        self.wa = wa
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
        if wa:
            self.linear = nn.Linear(32 * 7 * 7, self.classes * 2)
        else:
            self.linear = nn.Linear(32 * 7 * 7, self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (args.batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        logit = self.linear(x)
        # logit.shape [batch, 20]
        if self.wa:
            mu, sigma = logit.split(self.classes, 1)
            # mu.shape [batch, 10]; sigma.shape [batch, 10]
            return mu, sigma
        else:
            mu = logit
            return mu
