import torch.nn as nn
import torch.nn.functional as F

import cnn

def make_model(args):
    return Aleatoric(args)

class Aleatoric(nn.Module):
    def __init__(self, args):
        super(Aleatoric, self).__init__()

        self.drop_rate = args.drop_rate
        self.in_c = args.in_channels
        self.classes = args.classes
        self.n_feats = args.n_feats
        self.mode = args.mode
        self.model = nn.ModuleList()

        self.model.append(
            cnn.CNN(self.in_c,
                    self.n_feats,
                    self.drop_rate)
        )
        self.model.append(
            cnn.CNN(self.n_feats,
                    self.n_feats * 2,
                    self.drop_rate)
        )
        self.model.append(
            cnn.Flatten(self.n_feats * 2 * 7 * 7,
                        self.classes)
        )

        
