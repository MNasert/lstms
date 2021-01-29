import numpy as np
#TODO build Base
class LSTMCell:
    def __init__(self, insize: int, outsize: int, lr: float = 1e-4,recurrencies = 1, bias: bool = False):
        super(LSTMCell, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.lr = lr
        self.recurrencies = recurrencies
        self.bias = bias
        self.input_prev = None
        self.h_prev = None
        self.c_prev = None
        self.i_f_o_c_prev = None
        self.grd_i_f_o_c = None
        self.weigthmatrix = np.random.randn((self.insize+bias), self.outsize)
        self.forgetmatrix = np.random.randn((self.insize+bias), self.outsize)
        self.inputmatrix = np.random.randn((self.insize+bias), self.outsize)
        self.outputmatrix = np.random.randn((self.insize+bias), self.outsize)
        self.counter = 0

    def forward(self, x, h_prev=None, c_prev=None, counter=None):
        self.input_prev = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        if h_prev is None:
            h_prev = np.zeros(x.shape)
        if self.bias:
            x = np.vstack((x, 1))
            h_prev = np.vstack((h_prev, 1))
        if c_prev is None:
            c_prev = np.zeros(x.shape)

        i = self.sg((x.T if x.shape[1] != self.inputmatrix.shape[0] else x @ self.inputmatrix) + h_prev.T if h_prev.shape[1] != self.inputmatrix.shape[0] else h_prev @ self.inputmatrix)
        f = self.sg((x.T if x.shape[1] != self.inputmatrix.shape[0] else x @ self.forgetmatrix) + h_prev.T if h_prev.shape[1] != self.inputmatrix.shape[0] else h_prev @ self.forgetmatrix)
        o = self.sg((x.T if x.shape[1] != self.inputmatrix.shape[0] else x @ self.outputmatrix) + h_prev.T if h_prev.shape[1] != self.inputmatrix.shape[0] else h_prev @ self.outputmatrix)

        c_ = np.tanh(x.T if x.shape[1] != self.inputmatrix.shape[0] else x @ self.weigthmatrix + h_prev.T if h_prev.shape[1] != self.inputmatrix.shape[0] else h_prev  @ self.weigthmatrix)
        cf = c_prev * f
        ci = i * c_prev
        c = self.sg(cf+ci)
        h = np.tanh(c_) * o
        self.i_f_o_c_prev = [(self.i_f_o_c_prev[0]+i)/2, (self.i_f_o_c_prev[1] + f)/2, (self.i_f_o_c_prev[2] + o)/2, (self.i_f_o_c_prev[3] + c)/2] if self.i_f_o_c_prev is not None else [i, f, o, c]
        if counter is None:
            counter = 1
        else:
            counter += 1
        print(counter, h)
        if counter < self.recurrencies:
            x, h, c = self.forward(x, h, c, counter)
        return x, h, c

    #TODO move to Softmax-Layer ASAP
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # max(x) subtracted for numerical stability
        return e_x / np.sum(e_x)
    #############

    #TODO move to another class ASAP
    def sg(self, x):
        return 1/(1+np.exp(-x))

    def der_sg(self, x):
        return self.sg(x) * (1 - self.sg(x))

    def der_tanh(self, x):
        return 1-np.tanh(x)**2
    ############

    #TODO build optimizer (optional)
    def backward(self, grd_O, grd_C, grd_I, grd_F):
        grd_O = self.der_sg(np.tanh(self.i_f_o_c_prev[3]) @ grd_O) @ grd_O
        grd_C = np.tanh(self.i_f_o_c_prev[3]) * self.i_f_o_c_prev[2] * self.der_tanh(self.i_f_o_c_prev[3])
        grd_F = self.der_sg(self.i_f_o_c_prev[3] @ grd_F) @ grd_F
        grd_I = self.der_tanh(self.i_f_o_c_prev[3] * grd_C) * grd_C
        grd_C = self.der_sg(grd_C * self.i_f_o_c_prev[0]) * grd_C
        self.grd_i_f_o_c = [(self.grd_i_f_o_c[0] + grd_I) / 2, (self.grd_i_f_o_c[1] + grd_F) / 2,
                          (self.grd_i_f_o_c[2] + grd_O) / 2,
                          (self.grd_i_f_o_c[3] + grd_C) / 2] if self.grd_i_f_o_c is not None else [grd_I, grd_F, grd_O, grd_C]

        return grd_O, grd_C, grd_I, grd_F

    def step(self):
        self.inputmatrix = self.inputmatrix - self.lr*self.grd_i_f_o_c[0]
        self.forgetmatrix = self.forgetmatrix - self.lr*self.grd_i_f_o_c[1]
        self.outputmatrix = self.outputmatrix - self.lr*self.grd_i_f_o_c[2]
        self.weigthmatrix = self.weigthmatrix - self.lr*self.grd_i_f_o_c[3]
    ###########
    def get_args(self):
        return self.insize, self.outsize, self.lr, self.bias

    def get_params(self):
        return self.inputmatrix, self.forgetmatrix, self.outputmatrix, self.weigthmatrix
