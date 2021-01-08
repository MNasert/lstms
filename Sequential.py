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
        grdW = .5*(pred-y)**2
        grdH = .5*(pred-y)**2
        for i in reversed(range(len(self.layers))):
            grdW, grdH = self.layers[i].backward(grdW, grdH)

    def step(self):
        for i in reversed(range(len(self.layers))):
            self.layers[i].step()
