import torch


class Optimiser:
    def __init__(self, parameters, alpha=1e-3, l2_penalty=0, l1_penalty=0):
        self.parameters = parameters
        self.alpha = alpha  # Learning rate.
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty

    def compute_loss_gradient(self, parameter):
        """
        Compute gradient of loss function including l1 / l2 regularisation for parameter
         after back-propagating objective function.
        """
        return parameter.grad + self.l2_penalty * parameter.data + self.l1_penalty * parameter.data.sign()

    def compute_gradient(self, parameter):
        """Compute gradient for parameter update according to individual optimiser."""
        raise NotImplementedError

    def step(self):
        """Take optimisation step, updating all trainable parameters in the network."""
        for parameter in self.parameters:
            parameter_gradient = self.compute_gradient(parameter)
            parameter.data -= self.alpha * parameter_gradient

    def zero_grad(self):
        """Zero all gradients after optimisation step."""
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.data.zero_()


class SGD(Optimiser):
    def __init__(self, parameters, alpha=0.1, l1_penalty=0, l2_penalty=0):
        super(SGD, self).__init__(parameters, alpha, l1_penalty, l2_penalty)

    def compute_gradient(self, parameter):
        return self.alpha * self.compute_loss_gradient(parameter)


class Adam(Optimiser):
    def __init__(self, parameters, l1_penalty=0, l2_penalty=0, alpha=1e-3, beta_1=0.9, beta_2=0.99, eps=1e-8):
        super().__init__(parameters, alpha=alpha, l1_penalty=l1_penalty, l2_penalty=l2_penalty)
        # Adam hyper-parameters (see ADAM, Kingma & Ba, 2015)
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        # Store 1st and second moments for all parameters
        self.m_s = {parameter: 0.0 for parameter in parameters}
        self.v_s = {parameter: 0.0 for parameter in parameters}

    def compute_gradient(self, parameter):
        """Computing Adam parameter update step. (see Algorithm 1, ADAM, Kingma & Ba, 2015)"""
        alpha, b1, b2, eps = self.alpha, self.beta_1, self.beta_2, self.eps
        g_t = self.compute_loss_gradient(parameter)
        m = b1 * self.m_s[parameter] + (1 - b1) * g_t
        v = b2 * self.v_s[parameter] + (1 - b2) * g_t.pow(2)
        self.m_s[parameter], self.v_s[parameter] = m, v
        m_n = m / (1 - b1)
        v_n = v / (1 - b2)
        gradient_step = m_n / (eps + v_n.sqrt())
        return gradient_step


class AdamHD(Adam):
    """
    Hyper-gradient descent version of Adam.
    """
    def __init__(self, parameters, l1_penalty=0, l2_penalty=0,
                 alpha_0=1e-3, alpha_lr=1e-4, beta_1=0.9, beta_2=0.99, eps=1e-8):
        super().__init__(parameters, l1_penalty=l1_penalty, l2_penalty=l2_penalty,
                         alpha=alpha_0, beta_1=beta_1, beta_2=beta_2, eps=eps)
        self.alpha_lr = alpha_lr
        self.prev_gradients = {parameter: 0.0 for parameter in parameters}

    def compute_gradient(self, parameter):
        gradient = super(AdamHD, self).compute_gradient(parameter)
        prev_gradient = self.prev_gradients[parameter]
        self.alpha += self.alpha_lr * (gradient*prev_gradient).sum()
        self.prev_gradients[parameter] = gradient
        return gradient
