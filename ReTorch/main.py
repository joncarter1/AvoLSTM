from models import FCNN
from optim import SGD, Adam, AdamHD
import torch
import matplotlib.pyplot as plt
from losses import MSE


def g(x, noise=False):
    true_vals = torch.cos(x)+torch.sin(2*x)
    if not noise:
        return true_vals
    return true_vals + torch.normal(0, 0.1, x.shape)


if __name__ == '__main__':
    fcnn = FCNN()
    times = torch.linspace(-5, 5, 100)
    training_times = torch.linspace(-5, 5, 100).unsqueeze(-1)
    print(training_times.shape)
    training_values = g(training_times, True)
    values = g(times)
    optimiser = AdamHD(fcnn.parameters(), alpha_lr=1e-8)
    #optimiser = Adam(fcnn.parameters)
    epochs = 500

    for _ in range(epochs):
        predictions = fcnn(training_times)
        loss = MSE(training_values, predictions)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        print(f"MSE : {loss}")


    plotting = True

    if plotting:
        plt.figure()
        with torch.no_grad():
            for i in range(5):
                predictions = fcnn(torch.unsqueeze(times, -1), training=True).squeeze()
                if i == 0:
                    plt.plot(times, predictions, ls="-.", lw=0.5, color="red", label="NN draw")
                else:
                    plt.plot(times, predictions, ls="-.", lw=0.5, color="red")
            predictions = fcnn(torch.unsqueeze(times, -1), training=False).squeeze()
            plt.plot(times, predictions, label="Full NN", color="green")
        plt.plot(times, values, label="Ground truth function")
        plt.scatter(training_times, training_values, label="Measurements")

        #plt.ylim(-1, 1)
        plt.legend()
        plt.show()