import numpy as np

class Sequential:
    def __init__(self, layerlist, criterion, bias=False, lr=1e-4):
        super(Sequential, self).__init__()
        self.layers = layerlist
        self.lr = lr
        self.bias = bias
        self.criterion = criterion
        for i in layerlist:
            i.lr = self.lr

    def forward(self, x, hdn=None, c=None):
        for i in self.layers:
            x, hdn, c = i.forward(x, hdn, c)
        return hdn

    def backward(self, pred, y):
        grd = [self.criterion.loss(pred, y), self.criterion.loss(pred, y), self.criterion.loss(pred, y), self.criterion.loss(pred, y)]
        for i in reversed(range(len(self.layers))):
            grd = self.layers[i].backward(*grd)

    def step(self):
        for _ in range(len(self.layers)):
            self.layers[_].step()
