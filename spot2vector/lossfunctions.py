import torch


def zinb_loss(true, pred, pi, theta, eps=1e-10, reg_theta=1e-4):
    """
    Compute Zero-Inflated Negative Binomial loss.
    
    Parameters:
    - true: The true values.
    - pred: The predicted mean values.
    - pi: The predicted zero-inflation probabilities.
    - theta: The dispersion parameter of the negative binomial distribution.
    - eps: A small value used for numerical stability.
    
    Returns:
    - loss: The computed loss value.
    """
    # Negative Binomial
    t1 = torch.lgamma(theta + eps) + torch.lgamma(true + 1.0) - torch.lgamma(theta + true + eps)
    t2 = (theta + true) * torch.log(1.0 + (pred / (theta + eps))) + (true * (torch.log(theta + eps) - torch.log(pred + eps)))

    nb_loss = t1 + t2

    # Zero-Inflated
    zero_inflation = -torch.log(torch.where(true > 0, 1.0 - pi, pi) + eps)

    # add two part
    loss = zero_inflation + nb_loss * (1 - pi) + reg_theta * theta

    return loss.mean()
