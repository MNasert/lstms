import torch
from PyTorchStuff import LinearTestNet
from Modules import Archetypes
import numpy as np

def test_fc():
    x = np.array([[.1], [.2], [.3], [.4], [.5]])
    y = x*2

    PyTorch = LinearTestNet.LinNet(torch.nn.MSELoss(), insize=1, outsize=1, bias=False)
    NumPy = Archetypes.Linear(1, 1, lr=1e-2, bias=False)

    weightmat = PyTorch.fc1.weight.data
    NumPy.wm = np.array(weightmat[0])

    assert NumPy.forward(x[0]) == PyTorch.forward(torch.Tensor(x[0]))
    out, out_ = [], []
    loss_p_x_Numpy, loss_p_x_Pytorch = [], []
    optimizer = torch.optim.SGD(params=PyTorch.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    PyTorch.train(True)
    for epoch in range(500):
        loss_n = []
        for i in range(len(x)):
            out_np = NumPy.forward(x[i])
            delta = (y[i] - out_np)**2
            #y^2 - 2 y*o +o^2
            #-2y + 2o
            NumPy.backward(2*out_np - 2*y[i])
            NumPy.step()
            loss_n.append(delta)

            optimizer.zero_grad()
            out_py = PyTorch.forward(torch.Tensor(x[i]))
            loss = criterion(input=out_py, target=torch.Tensor(y[i]))
            loss.backward()
            optimizer.step()
            loss_n.append(loss.item())

        loss_p_x_Numpy.append(np.mean(delta))
        loss_n = []
        loss_p_x_Pytorch.append(np.mean(loss_n))
    for n in range(len(loss_p_x_Numpy)):
        epsilon = 0
        assert abs(loss_p_x_Numpy[n]-loss_p_x_Pytorch[n]) <= epsilon