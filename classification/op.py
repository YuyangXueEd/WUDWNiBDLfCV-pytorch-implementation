from torch.utils.tensorboard import SummaryWriter


class Operator:
    def __init__(self, config, check_point):
        self.config = config
        self.epochs = config.epochs
        self.mode = config.mode
        self.ckpt = check_point
        self.tensorboard = config.tensorboard

        if self.tensorboard:
            self.summary_writer = SummaryWriter(self.ckpt.log_dir, 300)

