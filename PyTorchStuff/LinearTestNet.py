import torch.nn as nn
import torch as t

class LinNet(nn.Module):
    def __init__(self, criterion, insize, outsize, lr=1e-4, bias=True, *args, **kwargs):
        super(LinNet, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.lr = lr
        self.criterion = criterion

        self.fc1 = nn.Linear(insize, outsize, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        return x







