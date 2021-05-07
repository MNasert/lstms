

class Sequential:
    def __init__(self, layers: list, lr=1e-2):
        super(Sequential, self).__init__()
        self.layers = layers
        self.lr = lr

    def forward(self, x):
        for i in self.layers:
            x = i.forward(x)
        return x

    def backward(self, loss):
        for i in reversed(self.layers):
            loss = i.backward(loss)

    def step(self):
        for i in self.layers:
            i.step()
