import time

from config import get_config

import data
from classification.model.aleatoric import aleatoric


def main(args):
    if args.data_name == 'MNIST':
        train_loader, test_loader = data.MNIST(args)

        if args.mode == 'aleatoric':
            start_time = time.time()
            best_acc = aleatoric(train_loader,
                                 test_loader,
                                 args)
            elapsed_time = time.time() - start_time
            print('Best test accuracy is: ', best_acc)  # 0.9918


if __name__ == "__main__":
    config = get_config()
    main(config)
