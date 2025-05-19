import torch
import torch.nn.functional as F

def compute_mrv_loss(activations, labels, predictions, major_regions, logger):
    device = activations.device
    total_loss = torch.tensor(0.0, device=device)
    num_samples = min(activations.size(0), labels.size(0))
    valid_samples = 0

    for i in range(num_samples):
        true_class = labels[i].item()
        pred_class = predictions[i].item()
        class_key = f"class_{true_class}"

        # Only apply if prediction is correct and MRV exists
        if pred_class == true_class and class_key in major_regions:
            try:
                mrv = torch.tensor(major_regions[class_key]["mrv"], device=device, dtype=activations.dtype)
                activation_i = activations[i]
                mse = F.mse_loss(activation_i, mrv)

                if torch.isfinite(mse):
                    total_loss += mse
                    valid_samples += 1
            except Exception as e:
                logger.warning(f"[MRV Skipped] Sample {i}: {e}")

    if valid_samples == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / valid_samples

def compute_hamming_loss(activations, labels, predictions, major_regions, k=10.0):
    device = activations.device
    total_loss = torch.tensor(0.0, device=device)
    valid_samples = 0

    for i in range(activations.size(0)):
        true_class = labels[i].item()
        pred_class = predictions[i].item()
        class_key = f"class_{true_class}"

        if pred_class == true_class and class_key in major_regions:
            try:
                mrv = torch.tensor(major_regions[class_key]["mrv"], device=device, dtype=activations.dtype)
                mrv_approx = k * torch.tanh(mrv)
                act_approx = k * torch.tanh(activations[i])
                euclidean_dist = F.mse_loss(act_approx, mrv_approx, reduction="mean")  # normalized per-dim
                total_loss += euclidean_dist
                valid_samples += 1
            except:
                continue

    return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=device)

def compute_rrv_loss(activations, labels, major_regions):
    """
    Computes the Relaxed Region Vector (RRV) Loss.
    Penalizes deviation from the RRV on selected feature dimensions using RDR mask.

    Args:
        activations (torch.Tensor): (batch_size, feature_dim)
        labels (torch.Tensor): (batch_size,)
        major_regions (dict): Dictionary containing RRV and RDR mask per class

    Returns:
        torch.Tensor: RRV loss value
    """
    device = activations.device
    total_loss = torch.tensor(0.0, device=device)
    num_samples = min(activations.size(0), labels.size(0))

    for i in range(num_samples):
        class_key = f"class_{labels[i].item()}"
        if class_key in major_regions:
            rrv = torch.tensor(major_regions[class_key]["rrv"], device=device, dtype=activations.dtype)
            rdr_mask = torch.tensor(major_regions[class_key]["rdr_mask"], device=device, dtype=activations.dtype)

            masked_act = activations[i] * rdr_mask
            total_loss += F.mse_loss(masked_act, rrv)

    return total_loss / num_samples if num_samples > 0 else torch.tensor(0.0, device=device)
