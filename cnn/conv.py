import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def con2d(X, Y):
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = X.reshape((1,1,6,8))
    Y = Y.reshape((1,1,6,7))
    lr = 3e-2

    for i in range(100):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if(i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')

    print(conv2d.weight.data.reshape((1, 2)))

if __name__ == '__main__':
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(f'X:{X}')
    K = torch.tensor([[1.0, -1.0]])
    print(f'K:{K}')
    Y = corr2d(X, K)
    print(f'Y:{Y}')
    con2d(X, Y)