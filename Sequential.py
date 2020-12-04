import numpy as np

class Sequential:
    def __init__(self, layerlist, lr):
        super(Sequential, self).__init__()
        self.layers = layerlist
        self.lr = lr
        for i in layerlist:
            i.lr = self.lr

    def forward(self, x, hdn=None):
        for i in self.layers:
            x, hdn = i.forward(x, hdn)
        return x

    def backward(self, pred, y):
        grdW = pred-y
        grdH = pred-y
        grdB = pred-y
        for i in reversed(range(len(self.layers))):
            grdW, grdH, grdB = self.layers[i].backward(grdW, grdH, grdB)

    def step(self):
        for i in reversed(range(len(self.layers))):
            self.layers[i].step()
