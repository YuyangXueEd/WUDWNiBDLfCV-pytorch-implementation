import os
import argparse
import time

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import datasets, transforms

from cnn import SimpleCNN


def aleatoric(train, test, args):
    device = torch.device('cuda:{}'.format(args.gpu)
               if torch.cuda.is_available() else 'cpu')

    cnn = SimpleCNN(classes=args.classes, wa=True).cuda()
    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)

    best_acc = 0
    start_time = time.time()
    elapsed_time = 0

    for epoch in range(epochs):
        cnn.train()

        for batch_idx, (train_x, train_y) in enumerate(train):
            # dropout can be used when training, since performing dropout also
            # when testing is the way to model epistemic uncertainty
            mu, sigma = cnn(train_x.cuda())

            prob_total = torch.zeros((samples, train_y.size(0), classes))
            for t in range(samples):
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
            for batch_idx, (test_x, test_y) in enumerate(test):
                test_mu, test_sigma = cnn(test_x.cuda())

                pred_y = torch.max(test_mu, 1)[1].cpu().data.numpy()
                correct += float((pred_y == test_y.data.numpy()).astype(int).sum())

                # Aleatoric uncertainty is measured by some function of test_sigma.

            accuracy = correct / float(len(test.dataset))
            print('-> Epoch: ', epoch, '| test accuracy: %.4f' % accuracy)
            if accuracy > best_acc:
                best_acc = accuracy

    return best_acc
    # elapsed_time = time.time() - start_time
    # print('Best test accuracy is: ', best_acc)  # 0.9918
    # print('Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
