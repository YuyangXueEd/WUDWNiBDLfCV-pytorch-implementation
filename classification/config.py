import argparse
import os

parser = argparse.ArgumentParser(description="Classification example")

# Environment
parser.add_argument("--tensorboard", type=bool, defalt=True)
parser.add_argument("--cpu", action='store_true', help="use cpu only")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--num_work", type=int, default=8)

# Data
parser.add_argument('--data_dir', type=str, default='../data/', help='the path of the dataset')
parser.add_argument('--data_name', type=str, default='MNIST', help='the name of the dataset')
parser.add_argument('--batch_size', type=int, default=512, help="Number per batch")

# Model
parser.add_argument('--mode', default='normal',
                    choices=['aleatoric', 'epistemic', 'combined', 'normal'], required=True,
                    help='which mode of uncertainty, choose from "aleatoric", "epistemic", "combined", and "normal"')
parser.add_argument('--drop_rate', type=float, default=0.2)
parser.add_argument('--classes', type=int, default=10, help="how many classes for the task")

# Train
parser.add_argument('--epochs', type=int, default=10, help="total training epochs")
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument("--decay", type=str, default='50-100-150-200')
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--optimizer", type=str, default='rmsprop',
                    choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--epsilon", type=float, default=1e-8)

# Test
parser.add_argument('--samples', type=int, default=10, help='number of samples')


def save_args(obj, defaults, kwargs):
    for k, v in defaults.iteritems():
        if k in kwargs:
            v = kwargs[k]

        setattr(obj, k, v)


def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    return config
