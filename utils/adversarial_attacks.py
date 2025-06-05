import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchattacks import AutoAttack

# ---------- helpers ---------------------------------------------------------

def _broadcast_bounds(t, mean, std):
    """
    Build per-channel lower/upper tensors that broadcast over `t`.
    Inputs/mean/std must all be on the same device.
    """
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)
    lower = (0.0 - mean) / std        # pixel = 0
    upper = (1.0 - mean) / std        # pixel = 1
    return lower.to(t), upper.to(t)


def _clip(x, lower, upper):
    return torch.max(torch.min(x, upper), lower)

# ---------- attacks ---------------------------------------------------------

def fgsm_attack(model, inputs, labels, *, epsilon, mean, std, **_):
    lower, upper = _broadcast_bounds(inputs, mean, std)

    inputs = inputs.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(inputs), labels)
    loss.backward()

    adv = inputs + epsilon * inputs.grad.sign()
    adv = _clip(adv, lower, upper)
    return adv.detach()


def bim_attack(model, inputs, labels, *, epsilon, alpha, num_iter, mean, std, **_):
    lower, upper = _broadcast_bounds(inputs, mean, std)
    adv = inputs.clone().detach()

    for _ in range(num_iter):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        delta = torch.clamp(adv - inputs, -epsilon, epsilon)
        adv = _clip(inputs + delta, lower, upper).detach()

    return adv


def pgd_attack(
    model, inputs, labels, *, epsilon, alpha, num_iter, restarts=5, mean, std
):
    lower, upper = _broadcast_bounds(inputs, mean, std)
    best_adv, best_loss = inputs, None

    for r in range(restarts):
        if r == 0:
            adv = inputs.clone().detach()
        else:                                    # random start
            noise = torch.empty_like(inputs).uniform_(-epsilon, epsilon)
            adv = _clip(inputs + noise, lower, upper)

        for _ in range(num_iter):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), labels)
            loss.backward()
            adv = adv + alpha * adv.grad.sign()
            delta = torch.clamp(adv - inputs, -epsilon, epsilon)
            adv = _clip(inputs + delta, lower, upper).detach()

        with torch.no_grad():
            cur_loss = F.cross_entropy(model(adv), labels)
            if best_loss is None or cur_loss > best_loss:
                best_loss, best_adv = cur_loss, adv.clone().detach()

    return best_adv

def cw_attack(model, inputs, labels, c=1e-4, kappa=0, max_iter=1000, lr=0.01):
    """Carlini & Wagner (C&W) Attack with parameters based on ICLR 2023 paper."""
    batch_size = inputs.size(0)
    device = inputs.device
    
    # Initialize perturbation variable
    w = torch.zeros_like(inputs, requires_grad=True, device=device)
    optimizer = optim.Adam([w], lr=lr)
    
    # Convert labels to one-hot encoding
    labels_one_hot = torch.zeros((batch_size, model(inputs).shape[1]), device=device)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    def loss_fn(modified_inputs, outputs):
        real = torch.sum(labels_one_hot * outputs, dim=1)
        other = torch.max((1 - labels_one_hot) * outputs - labels_one_hot * 10000, dim=1)[0]
        loss1 = torch.clamp(real - other + kappa, min=0).mean()
        loss2 = torch.norm(modified_inputs - inputs, p=2).mean()
        return loss1 + c * loss2
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        
        # Create adversarial example
        adv_inputs = torch.tanh(w + inputs)  # Enforce valid pixel range
        outputs = model(adv_inputs)
        loss = loss_fn(adv_inputs, outputs)
        
        loss.backward()
        optimizer.step()
    
    return torch.tanh(w + inputs).detach()

def autoattack(model, inputs, labels, dataset_name):
    """AutoAttack with dataset-specific epsilon values based on ICLR 2023 paper."""
    eps_dict = {"MNIST": 0.3, "F-MNIST": 0.1, "CIFAR-10": 0.0313}
    eps = eps_dict.get(dataset_name, 0.0313)  # Default to CIFAR-10 if dataset is unknown
    
    attacker = AutoAttack(model, norm='Linf', eps=eps, version="standard")
    adv_inputs = attacker(inputs, labels)
    return adv_inputs

