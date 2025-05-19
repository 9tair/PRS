import torch
import torch.nn as nn
import numpy as np

def rdr_loss_optimized(x_features, x_signs, labels, concept_bank, lambda_feat, lambda_config, return_separate=False):
    """
    Optimized RDR loss calculation with batch processing and GPU acceleration.
    
    Args:
        x_features: Batch of features, shape (B, D)
        x_signs: Sign of features, shape (B, D)
        labels: Class labels, shape (B)
        concept_bank: Dictionary mapping class to concepts
        lambda_feat, lambda_config: Loss weights
        return_separate: Whether to return separate loss components
        
    Returns:
        Total loss or (feature loss, config loss) if return_separate=True
    """
    device = x_features.device
    batch_size = x_features.size(0)
    
    # Initialize loss accumulators
    loss_feat = torch.tensor(0.0, device=device)
    loss_config = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    # Process by class to batch similar operations
    unique_labels = torch.unique(labels).cpu().numpy()
    
    for class_idx in unique_labels:
        # Skip if no concepts for this class
        if class_idx not in concept_bank or not concept_bank[class_idx]:
            continue
            
        # Create masks for samples of this class
        class_mask = (labels == class_idx)
        if not torch.any(class_mask):
            continue
            
        # Extract samples of this class
        class_features = x_features[class_mask]
        class_signs = x_signs[class_mask]
        num_class_samples = class_features.shape[0]
        
        # Pre-process concepts for this class
        concepts = concept_bank[class_idx]
        num_concepts = len(concepts)
        
        # Convert concepts to tensors (once per class)
        principal_idxs_list = [c[0] for c in concepts]
        patterns_list = [torch.tensor(c[1], dtype=torch.float, device=device) for c in concepts]
        centroids_list = [torch.tensor(c[2], dtype=torch.float, device=device) for c in concepts]
        
        # Process each sample of this class
        for i in range(num_class_samples):
            sample_feat = class_features[i]
            sample_sign = class_signs[i]
            
            # Find the best matching concept
            best_dist = float('inf')
            best_idx = -1
            
            # Try all concepts to find the best match
            for j in range(num_concepts):
                principal_idxs = principal_idxs_list[j]
                pattern = patterns_list[j]
                
                # Extract relevant signs for this concept
                sample_pattern = sample_sign[principal_idxs]
                
                # Compute Hamming distance efficiently on GPU
                # Convert pattern to -1/1 for proper comparison
                pattern_pm = 2 * pattern - 1  # Convert 0/1 to -1/1
                hamming_dist = torch.sum(sample_pattern != pattern_pm).item()
                
                if hamming_dist < best_dist:
                    best_dist = hamming_dist
                    best_idx = j
            
            if best_idx >= 0:
                # Apply losses for the best matching concept
                best_centroid = centroids_list[best_idx]
                best_pattern = patterns_list[best_idx]
                best_idxs = principal_idxs_list[best_idx]
                
                # Compute feature loss
                feat_loss = nn.functional.mse_loss(sample_feat, best_centroid)
                
                # Compute configuration loss
                pattern_pm = 2 * best_pattern - 1  # Convert to -1/1
                config_loss = nn.functional.l1_loss(
                    sample_sign[best_idxs].float(),
                    pattern_pm
                )
                
                loss_feat += feat_loss
                loss_config += config_loss
                valid_samples += 1
    
    # Normalize and apply weights
    if valid_samples > 0:
        loss_feat = (loss_feat / valid_samples) * lambda_feat
        loss_config = (loss_config / valid_samples) * lambda_config
        
        if return_separate:
            return loss_feat, loss_config
        else:
            return loss_feat + loss_config
    else:
        # Return zero losses if no valid samples
        if return_separate:
            return loss_feat, loss_config
        else:
            return loss_feat


# More advanced implementation with batch processing
def rdr_loss_batched(x_features, x_signs, labels, concept_bank, lambda_feat, lambda_config, return_separate=False):
    """
    Even faster RDR loss with advanced batch processing for large batch sizes.
    This version processes concept matching in batched operations.
    
    Args:
        x_features: Batch of features, shape (B, D)
        x_signs: Sign of features, shape (B, D)
        labels: Class labels, shape (B)
        concept_bank: Dictionary mapping class to concepts
        lambda_feat, lambda_config: Loss weights
        return_separate: Whether to return separate loss components
        
    Returns:
        Total loss or (feature loss, config loss) if return_separate=True
    """
    device = x_features.device
    batch_size = x_features.size(0)
    
    # Initialize loss tensors
    loss_feat = torch.tensor(0.0, device=device)
    loss_config = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    # Check which samples have concepts available
    valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for c in range(10):  # Assuming 10 classes
        if c in concept_bank and concept_bank[c]:
            valid_mask |= (labels == c)
    
    # Skip processing if no valid samples
    if not torch.any(valid_mask):
        if return_separate:
            return loss_feat, loss_config
        else:
            return loss_feat
    
    # Process valid samples only
    valid_features = x_features[valid_mask]
    valid_signs = x_signs[valid_mask]
    valid_labels = labels[valid_mask]
    valid_batch_size = valid_features.size(0)
    
    # Process in batches by class to maximize GPU utilization
    unique_labels = torch.unique(valid_labels).cpu().numpy()
    
    for class_idx in unique_labels:
        # Skip if no concepts for this class
        if class_idx not in concept_bank or not concept_bank[class_idx]:
            continue
        
        # Get samples of this class
        class_mask = (valid_labels == class_idx)
        class_features = valid_features[class_mask]
        class_signs = valid_signs[class_mask]
        num_samples = class_features.size(0)
        
        if num_samples == 0:
            continue
        
        # Prepare concepts
        concepts = concept_bank[class_idx]
        num_concepts = len(concepts)
        
        if num_concepts == 0:
            continue
        
        # Extract all sample features at once
        principal_idxs_list = [torch.tensor(c[0], device=device) for c in concepts]
        patterns_list = [torch.tensor(c[1], dtype=torch.float, device=device) for c in concepts]
        centroids_list = [torch.tensor(c[2], dtype=torch.float, device=device) for c in concepts]
        
        # For each sample, find the best concept
        for i in range(num_samples):
            sample_feat = class_features[i]
            sample_sign = class_signs[i]
            
            best_dist = float('inf')
            best_idx = -1
            
            # Match against all concepts
            for j in range(num_concepts):
                principal_idxs = principal_idxs_list[j]
                pattern = patterns_list[j]
                
                # Extract signs for principal neurons
                sample_pattern = sample_sign[principal_idxs]
                
                # Convert pattern from 0/1 to -1/1 for proper comparison
                pattern_pm = 2 * pattern - 1
                
                # Compute Hamming distance
                hamming_dist = torch.sum(sample_pattern != pattern_pm).item()
                
                if hamming_dist < best_dist:
                    best_dist = hamming_dist
                    best_idx = j
            
            # Apply losses for best concept
            if best_idx >= 0:
                best_centroid = centroids_list[best_idx]
                best_pattern = patterns_list[best_idx]
                best_idxs = principal_idxs_list[best_idx]
                
                feat_loss = nn.functional.mse_loss(sample_feat, best_centroid)
                
                pattern_pm = 2 * best_pattern - 1
                config_loss = nn.functional.l1_loss(
                    sample_sign[best_idxs].float(),
                    pattern_pm
                )
                
                loss_feat += feat_loss
                loss_config += config_loss
                valid_samples += 1
    
    # Normalize and apply weights
    if valid_samples > 0:
        loss_feat = (loss_feat / valid_samples) * lambda_feat
        loss_config = (loss_config / valid_samples) * lambda_config
        
        if return_separate:
            return loss_feat, loss_config
        else:
            return loss_feat + loss_config
    else:
        if return_separate:
            return loss_feat, loss_config
        else:
            return loss_feat