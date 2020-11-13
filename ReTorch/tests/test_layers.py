import torch
import torch.nn.functional as F
from ReTorch.layers import Conv1D, MaxPool1D
import numpy as np


def test_conv1d():
    cd = Conv1D(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
    a = torch.randn(5, 8).unsqueeze(1)
    output1 = cd(a)
    assert ((F.conv1d(a, cd.filter_bank, padding=1) - output1) < 1e-5).all()
    cd2 = Conv1D(in_channels=4, out_channels=10, kernel_size=3, padding=1, stride=1)
    output2 = cd2(output1)
    assert ((F.conv1d(output1, cd2.filter_bank, padding=1) - output2) < 1e-5).all()


def test_maxpool1d():
    kernel_size = np.random.randint(3, 10)
    padding = np.random.randint(0, np.floor(kernel_size/2))
    stride = np.random.randint(1, 10)
    mp = MaxPool1D(kernel_size=kernel_size, padding=padding, stride=stride)
    a = torch.randn(50, 80).unsqueeze(1)
    torch_mp = F.max_pool1d(input=a, kernel_size=kernel_size, padding=padding, stride=stride)
    assert (mp(a) - torch_mp < 1e-5).all()


if __name__ == "__main__":
    test_conv1d()
    test_maxpool1d()