import torch
import torch.nn as nn


class SymmetryLoss(nn.Module):
    def __init__(self):
        super(SymmetryLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x):
        x_mirrored = torch.flip(x, [-1])
        return self.criterion(x, x_mirrored)