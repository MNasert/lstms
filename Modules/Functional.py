import numpy as np


class Sigmoid:
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.ipt_prev = None

    def forward(self, x):
        self.ipt_prev = x
        return 1 / (1 + np.exp(-x))

    def backward(self, grd_in):
        grd_out = self.forward(grd_in) * (1-self.forward(grd_in))
        return grd_out

class TanH:
    def __init__(self):
        super(TanH, self).__init__()
        self.ipt_prev = None

    def forward(self, x):
        self.ipt_prev = x
        return np.tanh(x)

    def backward(self, grd_in):
        grd_out = 1 - self.forward(grd_in)**2
        return grd_out

class ReLU:
    def __init__(self):
        super(ReLU, self).__init__()
        self.ipt_prev = None

    def forward(self, x):
        self.ipt_prev = x
        return x * (x > 0)

    def backward(self, grd_in):
        grd_out = 1. * (grd_in > 0)
        return grd_out
