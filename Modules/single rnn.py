import numpy as np

w = np.random.rand(3,3)
wh = np.random.rand(3,3)
b = np.random.rand(1,3)
bh = np.random.rand(1,3)


w1 = np.random.rand(3,3)
wh1 = np.random.rand(3,3)
b1 = np.random.rand(1,3)
bh1 = np.random.rand(1,3)


w2 = np.random.rand(3,1)
wh2 = np.random.rand(3,1)
b2 = np.random.rand(1,1)
bh2 = np.random.rand(1,1)
w_s = [w, w1, w2]
wh_s = [wh, wh1, wh2]
b_s = [b,b1,b2]
bh_s = [bh, bh1, bh2]

x = np.random.rand(4,1,3)
y = np.random.rand(4,1,1)
def sigmoid(x, w, b):
    return 1/1+np.exp(-(w*x)+b)

def dsigmoid(x,w,b):
    sigmoid(x,w,b)*(1-sigmoid(x,w,b))
def forward(x, w_s, wh_s, b_s, bh_s, h = None):
    o = []
    if h is None:
        h = []
        o.append(sigmoid(x @ w_s[0] + b_s[0], w_s[0], b_s[0]))
        h.append(x @ wh_s[0] + bh_s[0])
    else:
        o.append(sigmoid((x+h[-1]) @ w_s[0] + b_s[0], w_s[0], b_s[0]))
        h.append((x+h[-1]) @ wh_s[0] + bh_s[0])


    o.append(sigmoid((x+h[-1]) @ w_s[1] + b_s[1], w_s[1], b_s[1]))
    h.append((x+h[-1]) @ wh_s[1] + bh_s[1])

    o.append((x+h[-1]) @ w_s[2] + b_s[2])
    h.append((x+h[-1]) @ wh_s[2] + bh_s[2])

    return o, h

o, h = forward(x[0], w_s, wh_s, b_s, bh_s)

def backward(x, o, y, w_s, wh_s, b_s, bh_s):
    gradw2 = (x-y)**2
    gradb2 = (x-y)**2

    gradh2 =

    gradw1 = gradw2 * w_s[0]
    gradb1 =

    gradh1 =
    gradw  = gradw1 * w_s[1]
    gradb  =

    gradh =

def step(gradw, gradwh, gradb, gradbh, w, wh, b, bh):
    w = w-gradw*.01
    wh = wh-gradwh*.01
    b = b-gradb*.01
    bh = bh-gradbh*.01
    return w, wh, b, bh