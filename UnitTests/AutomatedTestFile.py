from lstms.Modules.Rnncell import RNNcell
from lstms.Modules.Sequential import Sequential
from lstms.Modules.Functional import TanH, Sigmoid
import lstms.Modules.Crits as Crits
from lstms.Modules.GRUCell import GRUCell as GRUCell
from lstms.Modules.LSTMCell import LSTMCell as LSTMCell
import lstms.Modules.Optim as Optimizers
from sympy import *
import numpy as np
import sys as sys

for i in range(10):
    x = np.random.randint(1,10)
    y = np.random.randint(1,10)
    r = np.random.randint(1,5)
    lstm = LSTMCell(x,y, recurrencies=r)
    x_i = np.array(x*[[np.random.random()]])
    y_i = np.random.rand(y)
    x,pred,q=np.array(lstm.forward(x_i))
    if pred.shape != y_i.shape:
        print("LSTM-Shape-Error")
        raise AssertionError

    for i in range(10):
        x = np.random.randint(1, 10)
        y = np.random.randint(1, 10)
        r = np.random.randint(1, 5)
        lstm = LSTMCell(x, y, recurrencies=r)
        x_i = np.array(x * [[np.random.random()]])
        y_i = np.random.rand(y)
        x, pred, q = np.array(lstm.forward(x_i))
        if pred.shape != y_i.shape:
            print("GRU-Shape-Error")
            raise AssertionError

    for i in range(10):
        x = np.random.randint(1, 10)
        y = np.random.randint(1, 10)
        r = np.random.randint(1, 5)
        lstm = LSTMCell(x, y, recurrencies=r)
        x_i = np.array(x * [[np.random.random()]])
        y_i = np.random.rand(y)
        x, pred, q = np.array(lstm.forward(x_i))
        if pred.shape != y_i.shape:
            print("RNN-Shape-Error")
            raise AssertionError