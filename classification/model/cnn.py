import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d(in_c, out_c, kernel_size, bias=True):
    return nn.Conv2d(in_c,
                     out_c,
                     kernel_size,
                     padding=(kernel_size // 2),
                     bias=bias)


class CNN(nn.Module):
    def __init__(self, in_c, n_feats, drop_rate):
        super(CNN, self).__init__()
        self.in_c = in_c
        self.n_feats = n_feats
        self.drop_rate = drop_rate

        self.conv = nn.Sequential(
            conv2d(self.in_c, self.n_feats, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output (16, 14, 14)
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        x = self.conv(x)

class Flatten(nn.Module):
    def __init__(self, in_c, classes,  wa):
        super(Flatten, self).__init__()

        self.in_c = in_c
        self.classes = classes
        self.wa = wa

        if(self.wa):
            self.linear = nn.Linear(self.in_c, self.classes * 2)
        else:
            self.linear = nn.Linear(self.in_c, self.classes)

    def forward(self, x):
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
