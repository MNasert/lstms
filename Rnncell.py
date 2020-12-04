import numpy as np

class RNNcell:
    def __init__(self, insize: int, outsize: int, lr: float = 1e-4):
        super(RNNcell, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.lr = lr
        self.wChange = None
        self.hChange = None
        self.bChange = None
        self.input_prev = None
        self.wm = np.random.rand(self.insize, self.outsize)
        self.hm = np.random.rand(self.insize, self.outsize)
        self.bm = np.random.rand(1, self.outsize)

    def forward(self, x, hdn=None):
        self.input_prev = x
        x = np.transpose(self.wm) @ x
        hdn = x + (np.transpose(self.hm) @ self.input_prev) + np.transpose(self.bm)
        return x, hdn

    def backward(self, grd_inW, grd_inH, grd_inB):
        self.wChange = self.input_prev @ np.transpose(grd_inW)
        self.hChange = self.input_prev @ np.transpose(grd_inH)
        self.bChange = self.input_prev * np.transpose(grd_inB)
        grd_outW = self.wm @ grd_inW
        grd_outH = self.hm @ grd_inH
        grd_outB = self.bm * np.mean(grd_inB)
        return grd_outW, grd_outH, grd_outB

    def step(self):
        self.wm = self.wm - self.lr * self.wChange
        self.hm = self.hm - self.lr * self.hChange
        self.bm = self.bm - self.lr * self.bChange

    def get_args(self):
        return self.insize, self.outsize, self.lr
