import torch
import numpy as np


def MSE(predictions, labels):
    return ((predictions-labels).pow(2)).mean()


def gaussian_loss(means, log_vars, labels):
    """Negative log-likelihood under an isotropic Gaussian."""
    inv_vars = (-log_vars).exp()
    return 0.5*((labels-means).pow(2)*inv_vars).mean() + 0.5*log_vars.mean()
