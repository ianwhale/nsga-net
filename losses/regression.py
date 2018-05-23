# regression.py

import torch
from torch import nn

class Regression(nn.Module):

    def __init__(self):
        super(Regression, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        loss = self.loss.forward(input, target)

        return loss