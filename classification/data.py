import os

import torch.utils.data as data
from torchvision import datasets, transforms


def MNIST(args):
    if not os.path.exists(args.data) or not os.listdir(args.data):
        DOWNLOAD_MNIST = True
    else:
        DOWNLOAD_MNIST = False

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

    return train_loader, test_loader
