from lstms.Rnncell import RNNcell
from lstms.Sequential import Sequential
from lstms.Functional import TanH
import numpy as np
testnet = Sequential([
    RNNcell(2, 10),
    TanH(),
    RNNcell(10, 10),
    TanH(),
    RNNcell(10, 1),
    TanH()
], lr=1e-4)
targnet = Sequential([
    RNNcell(2, 10),
    TanH(),
    RNNcell(10, 10),
    TanH(),
    RNNcell(10, 1),
    TanH()
], lr=1e-4)

setsize = 10
x = [np.random.rand(2, 1) for i in range(setsize)]
y = [targnet.forward(x)]

for j in range(20000):
    c = None
    for i in range(len(x)):
        pred = testnet.forward(x[i])
        testnet.backward(pred, y[i])
        testnet.step()

