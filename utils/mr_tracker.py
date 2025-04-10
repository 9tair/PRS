from utils.logger import setup_logger
import numpy as np
import torch
import json
import os
from collections import defaultdict
from config import config 

def compute_major_regions(activations, labels, num_classes, logger):
    """
    Computes Major Region (MR), Extra Regions (ER), Relaxed Decision Region (RDR) masks, 
    and Relaxed Region Vector (RRV) for each class.

    Returns:
        dict: Per-class results with MR, ER, RDR, RRV, and added region label distributions.
        dict: Unique activation pattern to sample + label map.
    """
    RDR_AGREEMENT_THRESHOLD = config["rdr_agreement_threshold"]

    logger.info(f"Activations shape: {activations.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Number of classes: {num_classes}")

    class_patterns = {c: defaultdict(int) for c in range(num_classes)}
    unique_patterns = {}
    pattern_index_map = {}
    class_samples = {c: [] for c in range(num_classes)}

    binary_activations = np.sign(activations)
    binary_activations[binary_activations == 0] = -1

    pattern_counter = 0

    for idx, (pattern, label) in enumerate(zip(binary_activations, labels)):
        pattern_tuple = tuple(pattern)
        class_samples[label].append(idx)

        if pattern_tuple not in pattern_index_map:
            pattern_index_map[pattern_tuple] = pattern_counter
            unique_patterns[pattern_counter] = {
                "activation_pattern": list(pattern),
                "original_activations": [],
                "samples": [],
                "labels": []  # Track all labels in this pattern
            }
            pattern_counter += 1

        unique_idx = pattern_index_map[pattern_tuple]
        unique_patterns[unique_idx]["original_activations"].append(activations[idx].tolist())
        unique_patterns[unique_idx]["samples"].append(idx)
        unique_patterns[unique_idx]["labels"].append(int(label))  # Track label

        class_patterns[label][unique_idx] += 1

    results = {}

    for c in range(num_classes):
        if not class_patterns[c]:
            continue

        sorted_patterns = sorted(class_patterns[c].items(), key=lambda x: x[1], reverse=True)
        major_pattern_index, major_samples = sorted_patterns[0]

        # Extra regions with label distribution
        extra_regions = []
        for idx, count in sorted_patterns[1:]:
            region_labels = unique_patterns[idx]["labels"]
            labels, counts = np.unique(region_labels, return_counts=True)
            label_distribution = {int(k): int(v) for k, v in zip(labels, counts)}
            
            # FIXED: Verify that the sum of distribution counts equals the total samples in this pattern
            total_samples_in_pattern = len(unique_patterns[idx]["samples"])
            
            extra_regions.append({
                "activation_index": idx,
                "count": count,  # This is only the count for class c
                "total_pattern_samples": total_samples_in_pattern,  # This is the total number of samples with this pattern
                "class_distribution": label_distribution  # Distribution across all classes
            })

        major_region_indices = unique_patterns[major_pattern_index]["samples"]
        mrv = activations[major_region_indices].mean(axis=0)

        pattern_matrix = np.array([np.array(unique_patterns[idx]["activation_pattern"]) for idx, _ in sorted_patterns])
        sign_consistency = (np.mean(pattern_matrix == np.sign(np.sum(pattern_matrix, axis=0)), axis=0)) >= RDR_AGREEMENT_THRESHOLD
        classwise_rdr_mask = sign_consistency.astype(int)

        # New RRV computation based only on RDR mask and all class samples
        rrv = np.zeros_like(mrv)

        # Get all activations for samples of this class
        class_indices = class_samples[c]
        class_activations = activations[class_indices]  # Shape: (N_class_samples, feature_dim)

        # Only compute mean for dimensions where RDR mask is 1
        for dim_index, is_consistent in enumerate(classwise_rdr_mask):
            if is_consistent:
                rrv[dim_index] = class_activations[:, dim_index].mean()

        # Class distribution in major region
        major_region_labels = unique_patterns[major_pattern_index]["labels"]
        labels, counts = np.unique(major_region_labels, return_counts=True)
        major_distribution = {int(k): int(v) for k, v in zip(labels, counts)}
        
        # FIXED: Verify that the sum of distribution counts equals the total samples in the major pattern
        total_samples_in_major_pattern = len(unique_patterns[major_pattern_index]["samples"])

        results[f"class_{c}"] = {
            "num_total_samples": sum(class_patterns[c].values()),
            "num_decision_regions": len(class_patterns[c]),
            "major_region": {
                "count": major_samples,  # This is only the count for class c
                "total_pattern_samples": total_samples_in_major_pattern,  # This is the total count across all classes
                "activation_index": major_pattern_index,
                "class_distribution": major_distribution  # Distribution across all classes
            },
            "extra_regions": extra_regions,
            "mrv": mrv.tolist(),
            "rdr_mask": classwise_rdr_mask.tolist(),
            "rrv": rrv.tolist()
        }

    return results, unique_patterns

def save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, model_name, logger, prs_enabled=False, rrv_enabled=False, nan_enabled=False, warmup_epochs=50):
    """
    Saves major region statistics, activation pattern mappings, RDR masks, and Relaxed Region Vectors (RRV).

    Args:
        major_regions (dict): Major and extra region statistics.
        unique_patterns (dict): Unique activation pattern mappings.
        dataset_name (str): Dataset identifier.
        batch_size (int): Batch size used in training.
        model_name (str): Model identifier.
        prs_enabled (bool): Whether PRS regularization was used.
        warmup_epochs (int): Number of warmup epochs before applying PRS (if enabled).
    """
    suffix = f"_warmup_{warmup_epochs}_PRS" if prs_enabled else ""
    rrv = f"_RRV" if rrv_enabled else ""
    nan = f"_NAN" if nan_enabled else ""
    save_dir = f"models/saved/{model_name}_{dataset_name}_batch_{batch_size}{suffix}{rrv}{nan}"
    os.makedirs(save_dir, exist_ok=True)
    
    region_save_path = os.path.join(save_dir, "major_regions.json")
    pattern_save_path = os.path.join(save_dir, "activation_patterns.json")
    
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
    
    with open(region_save_path, "w") as f:
        json.dump(convert_to_serializable(major_regions), f, indent=4)
    logger.info(f"Major region statistics saved to {region_save_path}")
    
    with open(pattern_save_path, "w") as f:
        json.dump(convert_to_serializable(unique_patterns), f, indent=4)
    logger.info(f"Activation pattern mapping saved to {pattern_save_path}")