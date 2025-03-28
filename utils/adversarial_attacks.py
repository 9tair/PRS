import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchattacks import AutoAttack

def fgsm_attack(model, inputs, labels, epsilon):
    """FGSM attack applied on normalized images"""
    
    # Ensure requires_grad is enabled
    inputs.requires_grad = True

    # Forward pass
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Compute FGSM perturbation
    perturbation = epsilon * inputs.grad.sign()
    adv_inputs = inputs + perturbation

    # Clip to ensure valid normalized range
    adv_inputs = torch.clamp(adv_inputs, -1, 1)

    return adv_inputs.detach()


def bim_attack(model, inputs, labels, epsilon, alpha, num_iter):
    """BIM attack applied on normalized images"""
    
    adv_inputs = inputs.clone().detach()

    for _ in range(num_iter):
        adv_inputs.requires_grad = True

        # Forward pass
        outputs = model(adv_inputs)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Compute gradient sign
        grad_sign = adv_inputs.grad.sign()
        adv_inputs = adv_inputs + alpha * grad_sign

        # Project back to epsilon ball
        delta = torch.clamp(adv_inputs - inputs, -epsilon, epsilon)
        adv_inputs = torch.clamp(inputs + delta, -1, 1).detach()

    return adv_inputs


def pgd_attack(model, inputs, labels, epsilon, alpha, num_iter, restarts=1):
    """PGD attack applied on normalized images"""
    
    best_adv_inputs = inputs.clone().detach()
    best_loss = None

    for restart in range(restarts):
        # Start from a random point within the epsilon ball
        if restart > 0:
            random_noise = torch.empty_like(inputs).uniform_(-epsilon, epsilon)
            adv_inputs = torch.clamp(inputs + random_noise, -1, 1).detach()
        else:
            adv_inputs = inputs.clone().detach()
        
        for _ in range(num_iter):
            adv_inputs.requires_grad = True

            # Forward pass
            outputs = model(adv_inputs)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Compute PGD perturbation
            grad_sign = adv_inputs.grad.sign()
            adv_inputs = adv_inputs + alpha * grad_sign

            # Project back to epsilon ball
            delta = torch.clamp(adv_inputs - inputs, -epsilon, epsilon)
            adv_inputs = torch.clamp(inputs + delta, -1, 1).detach()

        # Keep the best adversarial example based on loss
        with torch.no_grad():
            final_outputs = model(adv_inputs)
            final_loss = F.cross_entropy(final_outputs, labels)

            if best_loss is None or final_loss > best_loss:
                best_loss = final_loss
                best_adv_inputs = adv_inputs.clone().detach()

    return best_adv_inputs



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

