import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model ...')

        self.is_train = args.is_train
        self.num_gpu = args.num_gpu
        self.mode = args.mode
        self.samples = args.samples
        module = import_module('model.' + self.mode)
        self.model = module.make_model(args).to(args.device)

    def save(self, ckpt, epoch):
        save_dirs = [os.path.join(ckpt.model_dir, self.mode, 'model_latest.pt'),
                     os.path.join(ckpt.model_dir, self.mode, 'model_{}.pt'.format(epoch))]

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, ckpt, cpu=False):
        epoch = ckpt.last_epoch
        kwargs = {}

        if cpu:
            # Load all tensors onto the CPU, using a function
            # torch.load('tensors.pt', map_location=lambda storage, loc: storage)
            kwargs = {'map_location': lambda storage, loc: storage}

        if epoch == -1:
            load_from = torch.load(os.path.join(ckpt.model_dir, self.mode, 'model_latest.pt'), **kwargs)
        else:
            load_from = torch.load(os.path.join(ckpt.model_dir, self.mode, 'model_{}.pt'.format(epoch)), **kwargs)

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward(self, x):
        if self.training:
            if self.num_gpu > 1:
                return P.data_parallel(self.model,
                                       x,
                                       list(range(self.num_gpu)))
            else:
                return self.model.forward(x)
        else:
            forward_func = self.model.forward
            if self.mode == 'normal':
                return forward_func(x)
            elif self.mode == 'aleatoric':
                return self.test_aleatoric(x, forward_func)
            elif self.mode == 'epistemic':
                return self.test_epistemic(x, forward_func)
            elif self.mode == 'combined':
                return self.test_combined(x, forward_func)

    @staticmethod
    def test_aleatoric(x, forward_func):
        results = forward_func(x)
        mean1 = results['mean']
        var1 = torch.exp(results['var'])
        var1_norm = var1 / var1.max()
        results = {'mean': mean1, 'var': var1_norm}
        return results
