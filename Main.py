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
sys.setrecursionlimit(9999)
model = Sequential([
    LSTMCell(2,2, recurrencies=4),
    ], criterion=Crits.MSELoss(), bias=True, lr=5e-3)
#TODO implement Linear layer -> _any_Cell(iptsize, _any_size) on Linear(_any_size, outsize)
x = np.array([[[.1], [.2]], [[.2], [.3]], [[.3], [.4]]])
y = np.array([x[1], x[2], [[.5], [.6]]])

for j in range(15):
    c = None
    pred = None
    for i in range(len(x)):
        pred = model.forward(x[i])
        model.backward(pred, y[i])
        model.step()
        print("pred:", pred, "\ny:", y[i], "\nx:", x[i])

#TODO rename _any_*Cell* to *Layer* as recurrencies are built
#For references:
import torch.optim