import numpy as np

class TanH:
    def __init__(self):
        super(TanH, self).__init__()

    def forward(self, x, hdn=None):
        return np.tanh(x), np.tanh(hdn) if hdn is not None else None

    def backward(self, grdW, grdH, grdB):
        grdW = 1-grdW**2
        grdH = 1-grdH**2
        grdB = 1-grdB**2
        return grdW, grdH, grdB

    def step(self):
        pass