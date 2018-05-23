# classification.py

import torch
from torch import nn

class Classification(nn.Module):

    def __init__(self):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.loss(input, target)
        return loss