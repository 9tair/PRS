from utils.logger import setup_logger
import numpy as np
import torch
import json
import os
from collections import defaultdict

def compute_major_regions(activations, labels, num_classes, logger):
    """
    Computes Major Region (MR) and Extra Regions (ER) for each class.

    Args:
        activations (numpy.ndarray): Original activation patterns of shape (num_samples, num_features).
        labels (numpy.ndarray): Ground-truth class labels of shape (num_samples,).
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary containing MR, ER, and summary statistics per class.
        dict: Dictionary mapping unique activation patterns to sample indices.
    """
    logger.info(f"Activations shape: {activations.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Number of classes: {num_classes}")
    
    print(f"Labels Shape: {labels.shape}, Activations Shape: {activations.shape}")
    print(f"Sample indices used: {[int(labels[i]) for i in range(num_classes)]}")

    # Track pattern occurrences (binarized for decision regions)
    class_patterns = {c: defaultdict(int) for c in range(num_classes)}
    unique_patterns = {}
    pattern_index_map = {}

    # Track **original** sample indices per class for MRV computation
    class_samples = {c: [] for c in range(num_classes)}

    # Convert activations to binary (-1, +1) **for decision region detection**
    binary_activations = np.sign(activations)
    binary_activations[binary_activations == 0] = -1  # Replace 0s with -1

    sample_tracker = set()
    pattern_counter = 0  

    for idx, (pattern, label) in enumerate(zip(binary_activations, labels)):
        pattern_tuple = tuple(pattern)

        # Store original sample indices per class
        class_samples[label].append(idx)

        if (idx, pattern_tuple) in sample_tracker:
            logger.warning(f"Duplicate mapping detected! Sample {idx} already assigned to pattern {pattern_tuple}")
        sample_tracker.add((idx, pattern_tuple))

        if pattern_tuple not in pattern_index_map:
            pattern_index_map[pattern_tuple] = pattern_counter
            unique_patterns[pattern_counter] = {
                "activation_pattern": list(pattern),
                "samples": []
            }
            pattern_counter += 1

        unique_patterns[pattern_index_map[pattern_tuple]]["samples"].append(idx)
        class_patterns[label][pattern_index_map[pattern_tuple]] += 1  

    # Compute MR, ER & statistics
    results = {}
    for c in range(num_classes):
        if not class_patterns[c]:  
            continue  

        sorted_patterns = sorted(class_patterns[c].items(), key=lambda x: x[1], reverse=True)
        major_pattern_index, major_samples = sorted_patterns[0]  

        extra_regions = [{"activation_index": idx, "count": count} for idx, count in sorted_patterns[1:]]

        # **Compute MRV using original activations (not binarized!)**
        mrv = activations[class_samples[c]].mean(axis=0).tolist()  

        expected_samples = (labels == c).sum()
        computed_samples = sum(class_patterns[c].values())
        if computed_samples != expected_samples:
            logger.warning(f"⚠️ Class {c} | Mismatch! Expected {expected_samples}, but found {computed_samples}")

        results[f"class_{c}"] = {
            "num_total_samples": computed_samples,
            "num_decision_regions": len(class_patterns[c]),
            "major_region": {
                "count": major_samples,
                "activation_index": major_pattern_index
            },
            "extra_regions": extra_regions,
            "mrv": mrv  
        }

        logger.info(f"Class {c} | Total Samples: {computed_samples} | Decision Regions: {results[f'class_{c}']['num_decision_regions']}")

    return results, unique_patterns

def save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, model_name, logger, prs_enabled=False, warmup_epochs=50):
    """
    Saves major region statistics and activation pattern mappings.

    Args:
        major_regions (dict): Major and extra region statistics.
        unique_patterns (dict): Unique activation pattern mappings.
        dataset_name (str): Dataset identifier.
        batch_size (int): Batch size used in training.
        model_name (str): Model identifier.
        prs_enabled (bool): Whether PRS regularization was used.
    """
    # Modify file paths based on PRS flag
    suffix = "_warmup_{warmup_epochs}_PRS" if prs_enabled else ""
    region_save_path = f"results/major_regions_{model_name}_{dataset_name}_batch_{batch_size}{suffix}.json"
    pattern_save_path = f"results/activation_patterns_{model_name}_{dataset_name}_batch_{batch_size}{suffix}.json"

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):  
            return obj.tolist()  
        elif isinstance(obj, (np.float16, np.float32, np.float64)):  
            return float(obj)  
        elif isinstance(obj, (np.int16, np.int32, np.int64)):  
            return int(obj)  
        elif isinstance(obj, tuple):  
            return list(obj)  
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj  

    # Save major regions
    with open(region_save_path, "w") as f:
        json.dump(convert_to_serializable(major_regions), f, indent=4)
    logger.info(f"Major region statistics saved to {region_save_path}")

    # Save activation pattern mapping
    with open(pattern_save_path, "w") as f:
        json.dump(convert_to_serializable(unique_patterns), f, indent=4)
    logger.info(f"Activation pattern mapping saved to {pattern_save_path}")
