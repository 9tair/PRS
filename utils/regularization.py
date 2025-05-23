import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

def compute_mrv_loss(
    activations: torch.Tensor,
    labels: torch.Tensor,
    major_regions: Dict[str, Any],
    logger: Any
) -> torch.Tensor:
    if activations is None or labels is None or major_regions is None or logger is None:
        return torch.tensor(0.0, device='cpu', requires_grad=True)

    if not isinstance(activations, torch.Tensor) or not isinstance(labels, torch.Tensor):
        logger.error("[MRV Loss] Activations or labels are not torch tensors.")
        return torch.tensor(0.0, device='cpu', requires_grad=True)

    device = activations.device
    batch_size, feature_dim = activations.shape

    total_loss = 0.0
    valid_sample_count = 0

    for i in range(batch_size):
        cls = labels[i].item()
        key = f"class_{cls}"
        mrv = major_regions.get(key, {}).get("mrv", None)

        if isinstance(mrv, (list, np.ndarray)) and len(mrv) == feature_dim:
            try:
                mrv_tensor = torch.tensor(mrv, device=device, dtype=torch.float32)

                if not torch.isfinite(mrv_tensor).all():
                    continue

                # Create a mask for positive MRV values
                positive_mask = mrv_tensor > 0

                # Apply the mask to both activation and MRV
                filtered_mrv = mrv_tensor[positive_mask]
                filtered_activation = activations[i][positive_mask]

                if filtered_mrv.numel() == 0:
                    continue  # No valid values to compute loss

                sample_loss = F.mse_loss(filtered_activation, filtered_mrv, reduction='mean')
                total_loss += sample_loss
                valid_sample_count += 1

            except Exception as e:
                logger.warning(f"[MRV Loss] Skipping class {cls}: {e}")
                continue

    if valid_sample_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / valid_sample_count

def compute_hamming_loss(activations: torch.Tensor, labels: torch.Tensor,
                         major_regions: Dict[str, Any], logger: Any) -> torch.Tensor:
    if activations is None or labels is None or major_regions is None or logger is None:
        return torch.tensor(0.0, device='cpu', requires_grad=True)

    if not isinstance(activations, torch.Tensor) or not isinstance(labels, torch.Tensor):
        logger.error("[Hamming Loss] Activations or labels are not torch tensors.")
        return torch.tensor(0.0, device='cpu', requires_grad=True)

    device = activations.device
    batch_size, feature_dim = activations.shape

    soft_activations = torch.tanh(activations)  # differentiable "soft sign"
    target_patterns = []

    valid_activations = []

    for i in range(batch_size):
        cls = labels[i].item()
        key = f"class_{cls}"
        mrv = major_regions.get(key, {}).get("mrv", None)

        if isinstance(mrv, (list, np.ndarray)) and len(mrv) == feature_dim:
            try:
                mrv_tensor = torch.tensor(mrv, device=device, dtype=torch.float32)
                mrv_sign = torch.sign(mrv_tensor).detach()  # Fixed reference pattern
                if not torch.isfinite(mrv_sign).all():
                    continue
                target_patterns.append(mrv_sign)
                valid_activations.append(soft_activations[i])
            except Exception as e:
                logger.warning(f"[Hamming Loss] Skipping class {cls}: {e}")
                continue

    if not valid_activations:
        return torch.tensor(0.0, device=device, requires_grad=True)

    act_batch = torch.stack(valid_activations)
    pattern_batch = torch.stack(target_patterns)
    return F.mse_loss(act_batch, pattern_batch)
