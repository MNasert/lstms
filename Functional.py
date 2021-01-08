import numpy as np

class TanH:
    def __init__(self):
        super(TanH, self).__init__()

    def forward(self, x, hdn=None):
        return np.tanh(x), np.tanh(hdn) if hdn is not None else None

    def backward(self, grdW, grdH):
        grdW = np.sinh(grdW)**2
        grdH = np.sinh(grdH)**2
        return grdW, grdH

    def step(self):
        pass