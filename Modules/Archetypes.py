import numpy as np

class Base:
    def __init__(self, insize, outsize, bias=True, lr=1e-2):
        super(Base, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.bias = bias
        self.lr = lr

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError
    #nimmt zugriff auf gradients und matrices -> TODO optimizer class
    def step(self, *args, **kwargs):
        raise NotImplementedError

    def __getinitargs__(self, *args, **kwargs):
        return self.insize, self.outsize, self.bias, self.lr

class Linear(Base):
    def __init__(self, insize, outsize, bias=True, lr=1e-2):
        Base.__init__(self, insize, outsize, bias, lr)
        self.wm = np.random.randn(outsize, insize)# shape: (2, 3)
        if bias:
            self.bm = np.random.rand(outsize, 1)
        self.grdW = None
        self.grdB = None
        self.iptPrev = None

    def forward(self, x):
        self.iptPrev = x.reshape(1, self.insize)# x= 3
        o = x @ self.wm.T# 1, 3  x 3, 2
        if self.bias:
            o = o + self.bm
        return o.reshape(1, self.outsize)

    def backward(self, grd, grdB):# grd = 1,2
        self.grdW = self.iptPrev.T @ grd#ipt prev = 1,3 -> 3,1 x 1,2 = 3,2.T = 2,3
        self.grdW = self.grdW
        # jede aktion eine Zeile
        #autograd.gradcheck pytorch
        grd_out = self.wm.T @ grd
        grd_out = grd_out
        if self.bias:
            grd_outB = (self.bm @ grdB.T).T
        else:
            grd_outB = grd
        """
        reason to use (wm @ grd.T).T:
        
        (x @ w.T).T= w.T.T @ x.T 
                   = w @ x.T 
        (x @ w.T)  = (w @ x.T).T
        """
        return grd_out, grd_outB

    def step(self):
        self.wm = self.wm - self.lr * self.grdW
        if self.bias:
            self.bm = self.bm - self.lr * self.grdB * self.bm


class RNNCell(Base):
    def __init__(self, insize, outsize, bias=True, lr=1e-2):
        Base.__init__(self, insize, outsize, bias, lr)
        super(RNNCell, self).__init__(insize, outsize, bias, lr)
        self.wm = np.random.randn(outsize, insize)  # shape: (2, 3)
        self.wh = np.random.randn(outsize, insize)  # shape: (2, 3)
        if bias:
            self.bm = np.random.rand(outsize, 1)
            self.bh = np.random.rand(outsize, 1)
        self.grdW = None
        self.grdWh = None
        self.grdB = None
        self.grdBh = None
        self.iptPrev = None
        self.hdn = None

    def forward(self, x, hdn=None):
        self.iptPrev = x
        if hdn is None:
            hdn = np.zeros_like(x)
        x = x + hdn
        o = x @ self.wm.T  # 1, 3  x 3, 2
        hdn = hdn @ self.wh.T
        self.hdn = hdn
        if self.bias:
            o = o + self.bm
            hdn = hdn + self.bh
        return o, hdn

    def backward(self, grdW, grdWh, grdB, grdBh):  # grd = 1,2
        self.grdW = (np.atleast_2d(self.iptPrev.T) @ np.atleast_2d(grdW)).T  # ipt prev = 1,3 -> 3,1 x 1,2 = 3,2.T = 2,3
        self.grdWh = (np.atleast_2d(self.hdn.T) @ np.atleast_2d(grdWh)).T  # ipt prev = 1,3 -> 3,1 x 1,2 = 3,2.T = 2,3
        self.grdB = grdB
        self.grdBh = grdBh
        grd_outW = (self.wm @ grdW.T).T
        grd_outWh = (self.wh @ grdWh.T).T
        if self.bias:
            grd_outB = (self.bm @ grdB.T).T
            grd_outBh = (self.bh @ self.grdBh.T).T
        else:
            grd_outB = grdB
            grd_outBh = grdBh
        """
        reason to use (wm @ grd.T).T:

        (x @ w^T).T= w.T.T @ x.T 
                   = w @ x.T 
        (x @ w.T)  = (w @ x.T).T
        """
        return grd_outW, grd_outWh, grd_outB, grd_outBh

    def step(self):
        self.wm = self.wm - self.lr * self.grdW
        self.wh = self.wh - self.lr * self.grdWh
        if self.bias:
            self.bm = self.bm - self.lr * self.grdB
            self.bh = self.bh - self.lr * self.grdBh


class RNNlayer:
    def __init__(self, insize, outsize, depth=3, bias=True, lr=1e-2):
        super(RNNlayer, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.depth = depth
        self.bias = bias
        self.lr = lr
        self.layers = []
        for i in range(depth - 1):
            self.layers.append(Linear(self.insize * 2, self.insize, bias=self.bias, lr=self.lr))
        self.layers.append(Linear(self.insize * 2, self.outsize, bias=self.bias, lr=self.lr))

    def forward(self, x):
        out = np.zeros_like(x)
        for i in self.layers:
            ipt = np.concatenate((x.flatten(), out.flatten()))
            #sigmoid bitte
            out = i.forward(ipt)

        return out

    def backward(self, grd):
        grdB = 0
        for i in reversed(self.layers):
            grd, grdB = i.backward(grd, grdB)
        #return

    def step(self):
        for i in self.layers:
            i.step()
