import torch
from layers import FC, Dropout, NNLayer
from activations import swish, linear, tanh
from torch import relu, sigmoid


class NN:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        """Find all attributes which are NN layers and add them to trainable parameters."""
        trainable_parameters = []
        for attr in dir(self):
            attr_val = getattr(self, attr)
            if isinstance(attr_val, NNLayer):
                trainable_parameters += attr_val.get_params()
        return trainable_parameters


class FCNN(NN):
    """Fully connected neural network"""
    def __init__(self):
        dropout = 0
        self.fc1 = FC(1, 50)
        self.fc2 = FC(50, 400)
        self.d1 = Dropout(dropout)
        self.fc3 = FC(400, 400)
        self.d2 = Dropout(dropout)
        self.fc4 = FC(400, 1)

    def forward(self, x, training=True, verbose=False):
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        h = self.d1(h)
        h = swish(self.fc3(h))
        h = self.d2(h)
        h = self.fc4(h)
        return h


class LSTM(NN):
    """LSTM neural network"""
    def __init__(self, in_size, h_size, out_size, y_layer1=100):
        self.w_i = FC(in_size+h_size, h_size)
        self.w_f = FC(in_size+h_size, h_size)
        self.w_o = FC(in_size+h_size, h_size)
        self.w_g = FC(in_size+h_size, h_size)
        self.w_y1 = FC(h_size, y_layer1)
        self.w_y2 = FC(y_layer1, out_size)

    def forward(self, x, h_, c_, training=True, verbose=False):
        stacked_state = torch.cat((x, h_), dim=1)
        i = sigmoid(self.w_i(stacked_state))
        f = sigmoid(self.w_f(stacked_state))
        o = sigmoid(self.w_o(stacked_state))
        g = tanh(self.w_g(stacked_state))
        c = f * c_ + i * g
        h = o * tanh(c)
        h_y = swish(self.w_y1(h))
        y = self.w_y2(h_y)
        return h, c, y
