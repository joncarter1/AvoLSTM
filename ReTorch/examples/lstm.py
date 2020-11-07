import torch
from optim import AdamHD, Adam
from models import LSTM
from losses import MSE
import matplotlib.pyplot as plt

def f(x):
    return torch.cos(x-0.4)+torch.sin(2*(x+0.2))

def g(t, noise=False):
    """Noisy mixture of sines function for 1D testing."""
    true_vals = torch.cos(t)+torch.sin(2*t)
    if not noise:
        return true_vals
    return true_vals + torch.normal(0, 0.1, t.shape)


if __name__ == "__main__":
    hidden_dim = 50
    dt = 0.1
    train_times = torch.arange(0, 10, dt)
    times = torch.arange(0, 40, dt)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(times, g(times), label="x = g(t)")
    ax[1].plot(times, f(g(times)), label="y=f(g(t))")
    plt.legend()
    plt.show()
    x_train = g(train_times).view(1, 1, -1).transpose(0, 2)
    x_test = g(times).view(1, 1, -1).transpose(0, 2)
    y_train = f(x_train).view(1, 1, -1).transpose(0, 2)
    sequence_length, batch_dim, x_dim = x_train.size()
    lstm = LSTM(x_dim, hidden_dim, 1)
    optimiser = AdamHD(lstm.parameters(), alpha_lr=1e-8)
    epochs = 100
    output_dim = 1

    def lstm_prediction(x):
        sequence_length = x.size(0)
        h = torch.zeros((batch_dim, hidden_dim))
        c = torch.zeros((batch_dim, hidden_dim))
        y_preds = torch.zeros((sequence_length, batch_dim, output_dim))
        for i, x_t in enumerate(x):
            h, c, y = lstm(x_t, h, c)
            y_preds[i] = y
        return y_preds

    for _ in range(epochs):
        y_preds = lstm_prediction(x_train)
        loss = MSE(y_train, y_preds)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        print(f"MSE : {loss}")

    with torch.no_grad():
        y_test_preds = lstm_prediction(x_test)
        plt.figure()
        plt.plot(times, y_test_preds.squeeze(), label="LSTM Prediction")
        plt.plot(times, f(g(times)), label="True function")
        plt.legend()
        plt.show()
