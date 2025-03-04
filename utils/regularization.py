import torch
import torch.nn.functional as F

def compute_mrv_loss(activations, labels, major_regions):
    """
    Computes the Major Region Variance (MRV) Loss.

    Args:
        activations (torch.Tensor): Model activations (batch_size, feature_dim).
        labels (torch.Tensor): True class labels (batch_size).
        major_regions (dict): Precomputed major region activations per class.

    Returns:
        torch.Tensor: MRV loss.
    """
    device = activations.device
    loss = 0.0
    
    # Ensure we use the correct batch size (minimum of activations and labels length)
    num_samples = min(activations.shape[0], labels.shape[0])

    # Preload MRVs as tensors (avoid modifying major_regions dictionary)
    precomputed_mrv = {class_id: torch.tensor(mr["mrv"], device=device) 
                        for class_id, mr in major_regions.items()}

    for i in range(num_samples):
        class_id = labels[i].item()
        if class_id in precomputed_mrv:
            loss += F.mse_loss(activations[i], precomputed_mrv[class_id])

    return loss / num_samples  # Normalize by batch size

def compute_hamming_loss(activations, labels, major_regions):
    """
    Computes the Hamming Distance Loss to reduce unnecessary activation patterns.

    Args:
        activations (torch.Tensor): Model activations (batch_size, feature_dim).
        labels (torch.Tensor): True class labels (batch_size).
        major_regions (dict): Precomputed major region activations per class.

    Returns:
        torch.Tensor: Hamming loss.
    """
    device = activations.device
    loss = 0.0
    
    # Ensure we use the correct batch size
    num_samples = min(activations.shape[0], labels.shape[0])

    # Convert activations to binary (-1, +1)
    binary_activations = torch.sign(activations)
    binary_activations[binary_activations == 0] = -1  # Replace 0s with -1

    # Precompute binary MRVs
    precomputed_mrv_bin = {class_id: torch.sign(torch.tensor(mr["mrv"], device=device)).int() 
                            for class_id, mr in major_regions.items()}

    for i in range(num_samples):
        class_id = labels[i].item()
        if class_id in precomputed_mrv_bin:
            hamming_dist = torch.sum((binary_activations[i].int() != precomputed_mrv_bin[class_id]).int()).float()
            loss += hamming_dist

    return loss / num_samples  # Normalize by batch size
