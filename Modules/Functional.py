import numpy as np
#TODO Linear class
class TanH:
    def __init__(self):
        super(TanH, self).__init__()
        self.y = None
        self.hdn = None
    def forward(self, x, hdn=None):
        return np.tanh(x), np.tanh(hdn) if hdn is not None else None

    def backward(self, grdW, grdH):
        grdW = np.sinh(grdW)**2
        grdH = np.sinh(grdH)**2
        return grdW, grdH

    def step(self):
        pass

class Sigmoid:
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.ipt_prev = None
    def forward(self, x, hdn=None):
        """
        :param x: input parameter
        :param hdn: (optional) hidden parameter
        :return: sigma(x)
            :math:
                1/(e^-x)
            :math:
        """
        self.y, self.hdn = 1/(1+np.exp(-x)), 1/(1+np.exp(-hdn))
        return self.y, self.hdn

    def backward(self, grdW, grdH):
        grdW = grdW*(self.y*(np.ones(self.y.shape)-self.y))
        grdH = grdH*(self.hdn*(np.ones(self.hdn.shape)-self.hdn))
        return grdW, grdH

    def step(self):
        pass


