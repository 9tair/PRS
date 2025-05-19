import torch
import torch.nn.functional as F
import torch.optim as optim
# numpy import was not used, so removed. If needed elsewhere, re-add.
from torchattacks import AutoAttack

def fgsm_attack(model, inputs, labels, epsilon):
    """FGSM attack applied on normalized images.
    'epsilon' is assumed to be scaled for the normalized input space.
    It can be a scalar or a tensor compatible with inputs for per-channel scaling.
    """
    # Ensure requires_grad is enabled on inputs for gradient computation
    inputs_clone = inputs.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(inputs_clone)
    loss = F.cross_entropy(outputs, labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Compute FGSM perturbation
    # inputs_clone.grad will not be None because requires_grad_(True) was called and it participated in loss.
    perturbation = epsilon * inputs_clone.grad.sign()
    adv_inputs = inputs_clone + perturbation

    # The perturbation is already at the L-infinity boundary defined by epsilon.
    # No further clamping of adv_inputs to a generic range like [-1, 1] is done here,
    # as 'inputs' are assumed to be in an arbitrary normalized space.
    return adv_inputs.detach()


def bim_attack(model, inputs, labels, epsilon, alpha, num_iter):
    """BIM (Basic Iterative Method) / I-FGSM attack applied on normalized images.
    'epsilon' and 'alpha' are assumed to be scaled for the normalized input space.
    """
    adv_inputs = inputs.clone().detach() # Start with original images

    for _ in range(num_iter):
        adv_inputs_iter = adv_inputs.clone().detach().requires_grad_(True)

        outputs = model(adv_inputs_iter)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad_sign = adv_inputs_iter.grad.sign()
        
        # Update adversarial example
        adv_inputs_step = adv_inputs_iter + alpha * grad_sign

        # Project perturbation back to L-infinity ball around original inputs
        # delta is the total perturbation from the original input
        delta = torch.clamp(adv_inputs_step - inputs, min=-epsilon, max=epsilon)
        adv_inputs = (inputs + delta).detach()

    return adv_inputs


def pgd_attack(model, inputs, labels, epsilon, alpha, num_iter, restarts=1):
    """PGD (Projected Gradient Descent) attack applied on normalized images.
    'epsilon' and 'alpha' are assumed to be scaled for the normalized input space.
    """
    max_loss = torch.full((1,), -float('inf'), device=inputs.device) # Use a tensor for device consistency
    best_adv_inputs = inputs.clone().detach()

    for i_restart in range(restarts):
        adv_inputs_restart = inputs.clone().detach()
        if i_restart > 0:
            # Start from a random point within the epsilon ball
            random_noise = torch.empty_like(inputs).uniform_(-1, 1) * epsilon # Ensure noise is within [-epsilon, epsilon]
            adv_inputs_restart = inputs + random_noise
            # Optionally, ensure adv_inputs_restart is within valid data range if known,
            # but for normalized data, L-inf ball is the main constraint here.

        current_adv_inputs = adv_inputs_restart
        for _ in range(num_iter):
            iter_adv_inputs = current_adv_inputs.clone().detach().requires_grad_(True)

            outputs = model(iter_adv_inputs)
            loss = F.cross_entropy(outputs, labels)

            model.zero_grad()
            loss.backward()

            grad_sign = iter_adv_inputs.grad.sign()
            adv_inputs_step = iter_adv_inputs + alpha * grad_sign

            # Project perturbation back to L-infinity ball around original inputs
            delta = torch.clamp(adv_inputs_step - inputs, min=-epsilon, max=epsilon)
            current_adv_inputs = (inputs + delta).detach()

        # After iterations for this restart, check loss
        with torch.no_grad():
            final_outputs = model(current_adv_inputs)
            # Using mean loss for the batch, as in the original script
            current_batch_loss = F.cross_entropy(final_outputs, labels) 
            if current_batch_loss > max_loss:
                max_loss = current_batch_loss
                best_adv_inputs = current_adv_inputs.clone().detach()
    
    return best_adv_inputs


def cw_attack(model, inputs, labels, c=1e-4, kappa=0, max_iter=1000, lr=0.01):
    """Carlini & Wagner L2 Attack on normalized images.
    'kappa' is the confidence margin: we want other_logits >= true_logit + kappa.
    'c' is the trade-off parameter for the L2 distortion.
    A small 'c' prioritizes finding an adversarial example (classification loss)
    over minimizing L2 distortion.
    """
    batch_size = inputs.size(0)
    num_classes = model(inputs).shape[1] # Get num_classes dynamically
    device = inputs.device

    # Perturbation variable, initialized to zeros
    delta = torch.zeros_like(inputs, requires_grad=True, device=device)
    optimizer = optim.Adam([delta], lr=lr)
    
    labels_one_hot = torch.zeros((batch_size, num_classes), device=device)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        
        adv_inputs = inputs + delta # Apply perturbation
        outputs = model(adv_inputs)
        
        true_logits = torch.sum(labels_one_hot * outputs, dim=1)
        # Set true class logits to a very small number to find max among others
        other_logits_prep = outputs.clone()
        other_logits_prep.masked_fill_(labels_one_hot.bool(), -float('inf'))
        max_other_logits = torch.max(other_logits_prep, dim=1)[0]

        # Classification loss: we want to minimize this term.
        # It becomes 0 when max_other_logits >= true_logits + kappa (successful untargeted attack with confidence kappa)
        # So, loss = max(0, true_logits - max_other_logits + kappa)
        # No, this is if we want true_logits to be smaller.
        # We want (max_other_logits) - true_logits >= kappa
        # So loss to minimize is max(0, -( (max_other_logits) - true_logits ) + kappa )
        # which is max(0, true_logits - max_other_logits + kappa) -- this makes true_logits smaller than (max_other - kappa)
        # Let's use: f(x') = max( Z(x')_t - max_{i!=t} Z(x')_i , -kappa ) based on many sources (target margin is -kappa)
        # Or equivalently (for minimization): max(0, (Z_t - max_other_Z) + kappa)
        # This tries to make Z_t - max_other_Z <= -kappa  => max_other_Z - Z_t >= kappa
        classification_loss_per_sample = torch.clamp(true_logits - max_other_logits + kappa, min=0)
        classification_loss = classification_loss_per_sample.sum() # Sum over batch

        # L2 distortion loss
        # Standard C&W often uses squared L2: l2_distortion = torch.sum(delta**2)
        # The original script used .mean() of L2 norm. Let's use sum of squared L2 norms.
        l2_distortion = torch.sum(delta**2) / batch_size # Mean of sum of squares per image, or just sum? Let's use sum of squares.
        # To match user's `loss2 = torch.norm(modified_inputs - inputs, p=2).mean()`
        # this would be `l2_distortion = torch.norm(delta.view(batch_size, -1), p=2, dim=1).mean()`
        # Let's stick to the typical C&W formulation which is sum of squares of delta.
        # Or, if we want to match user's original structure:
        # loss1 (classification) + c * loss2 (L2 norm)
        # If loss2 is mean L2 norm:
        l2_norm_terms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        l2_distortion_for_loss = l2_norm_terms.mean() # Mean of L2 norms

        # User's original formulation for total loss: classification_loss + c * l2_distortion_loss
        total_loss = classification_loss + c * l2_distortion_for_loss
        
        total_loss.backward()
        optimizer.step()
    
    final_adv_inputs = (inputs + delta).detach()
    # No generic clamping like [-1,1] as inputs are in an arbitrary normalized space.
    # If original data was [0,1] pixels, one might clamp final_adv_inputs to [0,1] here.
    return final_adv_inputs


def autoattack(model, inputs, labels, dataset_name):
    """AutoAttack wrapper.
    'model' should be prepared for AutoAttack (e.g., wrapped if it expects normalized inputs
    and AutoAttack works with [0,1] range inputs for its default eps).
    'inputs' should be in the range AutoAttack expects (typically [0,1]).
    'dataset_name' is used to fetch the standard epsilon for [0,1] space.
    """
    if "CIFAR10" in dataset_name.upper():
        processed_dataset_name = "CIFAR-10"
    elif "FMNIST" in dataset_name.upper() or "F-MNIST" in dataset_name.upper():
        processed_dataset_name = "F-MNIST"
    elif "MNIST" in dataset_name.upper():
        processed_dataset_name = "MNIST"
    else:
        processed_dataset_name = "CIFAR-10"
        print(f"Warning: Unknown dataset '{dataset_name}' for AutoAttack. Using CIFAR-10 epsilon as default.")

    eps_dict = {"MNIST": 0.3, "F-MNIST": 0.1, "CIFAR-10": 0.031372549}
    eps = eps_dict.get(processed_dataset_name, eps_dict["CIFAR-10"])
    
    print(f"AutoAttack: Using epsilon {eps} for dataset key '{processed_dataset_name}' (original: '{dataset_name}')")
    
    # Remove the 'device' argument from the AutoAttack constructor
    attacker = AutoAttack(model, norm='Linf', eps=eps, version="standard", verbose=False)
    # The attacker will use the device of the model and inputs.
    
    adv_inputs = attacker(inputs, labels)
    return adv_inputs.detach()