import numpy as np

class RNNcell:
    def __init__(self, insize: int, outsize: int, lr: float = 1e-4, bias: bool = True):
        super(RNNcell, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.lr = lr
        self.bias = bias
        self.wChange = None
        self.hChange = None
        self.bChange = None
        self.input_prev = None
        self.wm = np.random.rand(self.outsize, self.insize+bias)
        self.hm = np.random.rand(self.outsize, self.insize+bias)

    def forward(self, x, hdn=None):
        self.input_prev = x
        if self.bias:
            self.input_prev = x = np.vstack((self.input_prev, 1))
        q = self.input_prev + hdn if hdn is not None else np.ones(x.shape)
        q = self.wm @ q
        hdn = self.hm @ self.input_prev
        return q, hdn

    def backward(self, grd_inW, grd_inH):
        self.wChange = self.input_prev @ grd_inW.T if grd_inW.shape == self.input_prev.shape else grd_inW
        self.hChange = self.input_prev @ grd_inH.T if grd_inH.shape == self.input_prev.shape else grd_inH

        grd_outW = self.wm @ grd_inW.T if grd_inW.shape == self.wm.shape else grd_inW
        grd_outH = self.hm @ grd_inH.T if grd_inH.shape == self.hm.shape else grd_inH

        return grd_outW, grd_outH

    def step(self):
        self.wm = self.wm - np.mean(self.lr * self.wChange.T if self.wChange.shape == self.wm.shape else self.wChange)
        self.hm = self.hm - np.mean(self.lr * self.hChange.T if self.hChange.shape == self.hm.shape else self.hChange)

    def get_args(self):
        return self.insize, self.outsize, self.lr, self.bias

    def get_params(self):
        return self.wm, self.hm
