from utils.logger import global_logger  # Import from logger.py to avoid circular import
logger = global_logger  # Use the global logger

import numpy as np
import torch
import json
import os
from collections import defaultdict


def compute_major_regions(activations, labels, num_classes):
    """
    Computes Major Region (MR) and Extra Regions (ER) for each class efficiently.

    Args:
        activations (numpy.ndarray): Activation patterns of shape (num_samples, num_features).
        labels (numpy.ndarray): Ground-truth class labels of shape (num_samples,).
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary containing MR, ER, and summary statistics per class.
        dict: Dictionary mapping unique activation patterns to sample indices.
    """

    logger.info(f"Activations shape: {activations.shape}")  # Logs the shape of activations
    logger.info(f"Labels shape: {labels.shape}")  # Logs the shape of labels
    logger.info(f"Number of classes: {num_classes}")  # Logs the number of classes
    
    class_patterns = {c: defaultdict(int) for c in range(num_classes)}  # Track activation pattern counts
    unique_patterns = {}  # Stores activation patterns and their associated samples
    pattern_index_map = {}  # Maps activation pattern tuples to unique indices

    # Convert activations to binary (-1, +1)
    binary_activations = np.sign(activations)
    binary_activations[binary_activations == 0] = -1  # Replace 0s with -1

    # 🔹 DEBUG: Track activation assignments
    sample_tracker = set()  # Store (sample_index, pattern) to check for duplicates

    # Hash activation patterns & store count per class
    pattern_counter = 0  # Unique index counter
    for idx, (pattern, label) in enumerate(zip(binary_activations, labels)):
        pattern_tuple = tuple(pattern)  # Convert pattern to hashable tuple

        # Check for duplicate sample mapping
        if (idx, pattern_tuple) in sample_tracker:
            logger.warning(f"⚠️ Duplicate mapping detected! Sample {idx} already assigned to pattern {pattern_tuple}")
        sample_tracker.add((idx, pattern_tuple))

        if pattern_tuple not in pattern_index_map:
            pattern_index_map[pattern_tuple] = pattern_counter
            unique_patterns[pattern_counter] = {
                "activation_pattern": list(pattern),  # Store activation vector
                "samples": []
            }
            pattern_counter += 1
        
        unique_patterns[pattern_index_map[pattern_tuple]]["samples"].append(idx)
        class_patterns[label][pattern_index_map[pattern_tuple]] += 1  # Increment count

    # Compute MR, ER & statistics
    results = {}
    for c in range(num_classes):
        if not class_patterns[c]:  
            continue  # Skip if no activations

        # Sort by frequency (most common pattern = MR)
        sorted_patterns = sorted(class_patterns[c].items(), key=lambda x: x[1], reverse=True)
        major_pattern_index, major_samples = sorted_patterns[0]

        # Extra Regions (ER) → All other patterns
        extra_regions = [{"activation_index": idx, "count": count} for idx, count in sorted_patterns[1:]]

        # Compute MRV (Mean Activation Vector for MR)
        major_region_indices = unique_patterns[major_pattern_index]["samples"]
        mrv = activations[major_region_indices].astype(np.float32).mean(axis=0).tolist()  # Ensure float32

        # 🔹 DEBUG: Verify the number of samples per class
        expected_samples = (labels == c).sum()  # Count of samples in this class
        computed_samples = sum(class_patterns[c].values())
        if computed_samples != expected_samples:
            logger.warning(f"⚠️ Class {c} | MISMATCH! Expected {expected_samples}, but found {computed_samples}")

        results[f"class_{c}"] = {
            "num_total_samples": computed_samples,  # Total samples in class (should be 5000)
            "num_decision_regions": len(class_patterns[c]),  # Unique activation patterns in class
            "major_region": {
                "count": major_samples,  # MR Sample Count
                "activation_index": major_pattern_index  # MR Activation Index
            },
            "extra_regions": extra_regions,  # Store only sample counts
            "mrv": mrv  # Mean activation vector for MR
        }

        logger.info(f"Class {c} | Total Samples: {computed_samples} | Decision Regions: {results[f'class_{c}']['num_decision_regions']}")

    return results, unique_patterns


def save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, model_name):
    """Save major region statistics in a compressed format, including model name.
        Saves two files: one with statistics, and one with activation pattern mappings.
    """
    
    # Define save paths
    region_save_path = f"results/major_regions_{model_name}_{dataset_name}_batch_{batch_size}.json"
    pattern_save_path = f"results/activation_patterns_{model_name}_{dataset_name}_batch_{batch_size}.json"

    # Convert to JSON serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):  
            return obj.tolist()  # Convert arrays & tensors to lists
        elif isinstance(obj, (np.float16, np.float32, np.float64)):  
            return float(obj)  # Convert NumPy floats
        elif isinstance(obj, (np.int16, np.int32, np.int64)):  
            return int(obj)  # Convert NumPy ints
        elif isinstance(obj, tuple):  
            return list(obj)  # Convert tuples to lists
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj  # Keep other types as is

    # Save major regions
    with open(region_save_path, "w") as f:
        json.dump(convert_to_serializable(major_regions), f, indent=4)
    logger.info(f"Major region statistics saved to {region_save_path}")

    # Save activation pattern mapping
    with open(pattern_save_path, "w") as f:
        json.dump(convert_to_serializable(unique_patterns), f, indent=4)
    logger.info(f"Activation pattern mapping saved to {pattern_save_path}")
