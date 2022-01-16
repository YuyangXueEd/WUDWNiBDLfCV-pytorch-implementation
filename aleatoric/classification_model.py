import os
import argparse
import time

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="Aleatoric example")
parser.add_argument('--cuda', action='store_true', help='Use GPU')
parser.add_argument('--data', type=str, default='./data/', help='the path of the dataset')
parser.add_argument('--batch_size', type=int, default=512, help="Number per batch")
parser.add_argument('--classes', type=int, default=10, help="how many classes for the task")
parser.add_argument('--epochs', type=int, default=10, help="total training epoches")
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--samples', type=int, default=10, help='number of samples')
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


cnn = SimpleCNN().cuda()

print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)

best_acc = 0
start_time = time.time()
elapsed_time = 0

for epoch in range(args.epochs):
    cnn.train()

    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        # dropout can be used when training, since perfoming dropout also
        # when testing is the way to model epistemic uncertainty
        mu, sigma = cnn(train_x.cuda())

        prob_total = torch.zeros((args.samples, train_y.size(0), args.classes))
        for t in range(args.samples):
            # assume that each logit value is drawn from Gaussian distribution,
            # therefore the whole logit vector is drawn from multi-dimensional Gaussian distribution
            epsilon = torch.randn(sigma.size()).cuda()
            logit = mu + torch.mul(sigma, epsilon)
            prob_total[t] = F.softmax(logit, dim=1)

        prob_ave = torch.mean(prob_total, 0)
        loss = F.nll_loss(torch.log(prob_ave), train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: ', epoch, '| batch: ', batch_idx, '| train loss: %.4f' % loss.cpu().data.numpy())

        cnn.eval()
        correct = 0
        for batch_idx, (test_x, test_y) in enumerate(test_loader):
            test_mu, test_sigma = cnn(test_x.cuda())

            pred_y = torch.max(test_mu, 1)[1].cpu().data.numpy()
            correct += float((pred_y == test_y.data.numpy()).astype(int).sum())

            # Aleatoric uncertainty is measured by some function of test_sigma.

        accuracy = correct / float(len(test_loader.dataset))
        print('-> Epoch: ', epoch, '| test accuracy: %.4f' % accuracy)
        if accuracy > best_acc:
            best_acc = accuracy

elapsed_time = time.time() - start_time
print('Best test accuracy is: ', best_acc)   # 0.9918
# print('Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))