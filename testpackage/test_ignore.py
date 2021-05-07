from lstms.Modules.Archetypes import *
import matplotlib.pyplot as plt
layerA = RNNlayer(1, 1, 3, False)
x = np.array([[1], [0.5], [0.25]]).reshape(3, 1)
y = np.array([[3], [1.5], [0.75]]).reshape(3, 1)
h = None
lsl = []
for epoch in range(1):
    ls = 0
    for i in range(len(x)):
        o = layerA.forward(x[i])
        print(o)
        loss = (y[i]-o)**2
        layerA.backward(loss)
        layerA.step()


