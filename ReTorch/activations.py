import torch
from torch import tanh


def linear(x):
    return x


def swish(x):
    return x*torch.sigmoid(x)


def leaky_relu(alpha=0.1):
    def activation(x):
        return torch.max(x, alpha*x)
    return activation
