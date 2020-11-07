import torch
import torch.nn.functional as F


class NNLayer:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_params(self):
        return []


class FC(NNLayer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w = torch.randn((input_dim, output_dim))*torch.tensor(2.0/input_dim).sqrt()  # He initialisation
        self.w.requires_grad = True
        self.b = torch.randn((1, output_dim), requires_grad=True)

    def forward(self, x, training=True):
        return x@self.w + self.b

    def get_params(self):
        return [self.w, self.b]


class Dropout(NNLayer):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, training=True):
        """Apply dropout to rows of data matrix. Equivalent to turning off neuron."""
        if not training:
            return x
        dropout_probs = (1-self.p) * torch.ones(x.size(0))
        dropout_mask = torch.bernoulli(dropout_probs).long()
        output = (x.T * dropout_mask).T
        return output


class Conv1D(NNLayer):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_bank = torch.ones((out_channels, in_channels, kernel_size))
        self.filter_bank[1] = 0.0
        self.filter_bank.requires_grad = True
        self.padding = padding
        self.stride = stride

    def forward2(self, x, training=True):
        batch_size, x_dim = x.size()
        pad_vector = torch.zeros((batch_size, self.padding))
        padded_x = torch.cat((pad_vector, x, pad_vector), dim=1)
        stacked_x = padded_x.unsqueeze(0).repeat(self.out_channels, 1, 1)
        input_shape = x_dim + 2 * self.padding
        output_shape = int((x.size(1)-self.kernel_size+2*self.padding)/self.stride + 1)
        filter_tensor = torch.zeros((self.out_channels, input_shape, output_shape))
        for col_idx in range(output_shape):
            filter_tensor[:, self.stride*col_idx:self.stride*col_idx+self.kernel_size, col_idx] = self.filter_bank
        return torch.matmul(stacked_x, filter_tensor).transpose(0, 1)

    def forward(self, x, training=True):
        batch_size, x_channels, x_dim = x.size()
        assert x_channels == self.in_channels
        pad_vector = torch.zeros((batch_size, x_channels, self.padding))
        padded_x = torch.cat((pad_vector, x, pad_vector), dim=2)
        stacked_x = padded_x  #.transpose(0, 1)
        input_shape = x_dim + 2 * self.padding
        output_shape = int((x.size(-1)-self.kernel_size+2*self.padding)/self.stride + 1)
        filter_tensor = torch.zeros((self.out_channels, self.in_channels, input_shape, output_shape))
        for col_idx in range(output_shape):
            filter_tensor[:, :, self.stride*col_idx:self.stride*col_idx+self.kernel_size, col_idx] = self.filter_bank
        output = torch.tensordot(stacked_x, filter_tensor, dims=([1, 2], [1, 2]))
        return output

    def get_params(self):
        return self.filters


class MaxPool1D(NNLayer):
    def __init__(self, kernel_size, padding, stride):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, x, training=True):
        output_shape = int((x.size(2) - self.kernel_size + 2 * self.padding) / self.stride + 1)
        batch_size, channels, x_dim = x.size()

        if self.padding:
            pad_vector = -1e1000*torch.ones((batch_size, channels, self.padding))
            padded_x = torch.cat((pad_vector, x, pad_vector), dim=2)
        else:
            padded_x = x

        pooling_tensor = torch.zeros((batch_size, channels, output_shape, self.kernel_size))
        for i in range(self.kernel_size):
            indices = i + torch.arange(0, output_shape) * self.stride
            x_slice = padded_x[:, :, indices]
            pooling_tensor[:, :, :, i] = x_slice
        max_pool_tensor, _ = torch.max(pooling_tensor, dim=-1)
        return max_pool_tensor
