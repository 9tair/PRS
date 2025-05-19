import numpy as np
from collections import Counter

def greedy_select_principal_neurons(positive_bin, negative_bin, t):
    """Select t neurons with highest activation frequency gap between sets."""
    freq_pos = positive_bin.mean(axis=0)
    freq_neg = negative_bin.mean(axis=0)
    gaps = np.abs(freq_pos - freq_neg)
    return np.argsort(gaps)[-t:]

def build_rdr_concepts(activations, labels, num_classes=10, k=10, t=12, max_seeds=100):
    """
    Optimized version of RDR concept discovery with better numpy vectorization.
    
    Args:
        activations: Feature activations, shape (N, D)
        labels: Class labels, shape (N,)
        num_classes: Number of classes
        k: Number of neighbors for each seed
        t: Number of principal neurons to select
        max_seeds: Maximum number of seeds per class
        
    Returns:
        Dictionary mapping class index to list of (neuron_idxs, pattern, centroid)
    """
    # Pre-compute signs once
    signs = np.sign(activations)
    signs[signs == 0] = -1  # ensure no zeros
    
    # Initialize concept bank
    concept_bank = {c: [] for c in range(num_classes)}
    
    # Process each class (possibly in parallel)
    for c in range(num_classes):
        # Get indices for this class
        class_idxs = np.where(labels == c)[0]
        if len(class_idxs) == 0:
            continue
            
        # Select limited number of seeds
        seed_count = min(max_seeds, len(class_idxs))
        seeds = np.random.choice(class_idxs, size=seed_count, replace=False)
        
        # Get configurations for all seeds at once
        seed_configs = signs[seeds]
        
        # Extract class configurations once
        class_configs = signs[class_idxs]
        
        for i, seed_idx in enumerate(seeds):
            seed_config = seed_configs[i]
            
            # Compute all distances at once (vectorized Hamming distance)
            # This avoids the loop over all class samples
            dists = np.sum(class_configs != seed_config, axis=1)
            
            # Get top-k nearest neighbors
            neighbor_indices = np.argsort(dists)[:k]
            neighbor_idxs = class_idxs[neighbor_indices]
            
            # Extract semantic configurations
            semantic_configs = signs[neighbor_idxs]
            
            # Compute probability and frequency gap
            prob = np.mean(semantic_configs, axis=0)
            
            # Use boolean indexing to avoid creating large arrays
            mask = ~np.isin(np.arange(len(signs)), neighbor_idxs)
            neg_configs = signs[mask]
            
            # Compute frequency gap more efficiently
            neg_mean = np.mean(neg_configs, axis=0)
            freq_gap = np.abs(prob - neg_mean)
            
            # Find deterministic neurons
            C = np.where((prob == 1.0) | (prob == -1.0))[0]
            if len(C) == 0:
                continue
                
            # Select principal neurons
            idx_sorted = np.argsort(freq_gap[C])[::-1]
            selected_count = min(t, len(idx_sorted))
            selected_neurons = C[idx_sorted[:selected_count]]
            
            # Compute pattern and centroid
            selected_pattern = (prob[selected_neurons] > 0).astype(np.int32)
            centroid = np.mean(activations[neighbor_idxs], axis=0)
            
            # Store concept
            concept_bank[c].append((selected_neurons, selected_pattern, centroid))
    
    return concept_bank

