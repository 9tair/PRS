import torch
import torch.nn.functional as F

def fgsm_attack(model, inputs, labels, epsilon):
    """Fast Gradient Sign Method (FGSM)"""
    if epsilon == 0:
        return inputs  # Return clean samples if ε = 0
    
    model.eval()
    adv_inputs = inputs.clone().detach().requires_grad_(True)

    outputs = model(adv_inputs)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    perturbation = epsilon * adv_inputs.grad.sign()
    adv_inputs = adv_inputs + perturbation
    adv_inputs = torch.clamp(adv_inputs, 0, 1)

    return adv_inputs.detach()


def bim_attack(model, inputs, labels, epsilon, alpha, num_iter):
    """Basic Iterative Method (BIM)"""
    if epsilon == 0:
        return inputs  # Return clean samples if ε = 0

    model.eval()
    adv_inputs = inputs.clone().detach()

    for _ in range(num_iter):
        adv_inputs.requires_grad = True

        outputs = model(adv_inputs)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        perturbation = alpha * adv_inputs.grad.sign()
        adv_inputs = adv_inputs.detach() + perturbation
        adv_inputs = torch.min(torch.max(adv_inputs, inputs - epsilon), inputs + epsilon)
        adv_inputs = torch.clamp(adv_inputs, 0, 1)

    return adv_inputs.detach()


def pgd_attack(model, inputs, labels, epsilon, alpha, num_iter):
    """Projected Gradient Descent (PGD)"""
    if epsilon == 0:
        return inputs  # Return clean samples if ε = 0

    model.eval()
    adv_inputs = inputs.clone().detach() + (torch.rand_like(inputs) * 2 * epsilon - epsilon)
    adv_inputs = torch.clamp(adv_inputs, 0, 1)

    for _ in range(num_iter):
        adv_inputs.requires_grad = True

        outputs = model(adv_inputs)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        perturbation = alpha * adv_inputs.grad.sign()
        adv_inputs = adv_inputs.detach() + perturbation
        adv_inputs = torch.min(torch.max(adv_inputs, inputs - epsilon), inputs + epsilon)
        adv_inputs = torch.clamp(adv_inputs, 0, 1)

    return adv_inputs.detach()
