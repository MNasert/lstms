from lstms.Rnncell import RNNcell
from lstms.Sequential import Sequential
from lstms.Functional import TanH
import numpy as np
testnet = Sequential([
    RNNcell(2, 10, bias=False),
    TanH(),
    RNNcell(10, 1, bias=False),
    TanH()
], lr=1e-2)

x = np.array([[[.1], [.2]], [[.2], [.3]], [[.3], [.4]]])
y = np.array([[.3], [.4], [.5]])

for j in range(150):
    c = None
    for i in range(len(x)):
        pred = testnet.forward(x[i])
        print("pred:", pred, "\ny:", y[i], "\nx:", x[i])
        testnet.backward(pred, y[i])
        testnet.step()

