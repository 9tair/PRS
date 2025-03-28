from utils.logger import setup_logger
import numpy as np
import torch
import json
import os
from collections import defaultdict
from config import config  # Import config dictionary

def compute_major_regions(activations, labels, num_classes, logger):
    """
    Computes Major Region (MR), Extra Regions (ER), Relaxed Decision Region (RDR) masks, 
    and Relaxed Region Vector (RRV) for each class.

    Args:
        activations (numpy.ndarray): Original activation patterns of shape (num_samples, num_features).
        labels (numpy.ndarray): Ground-truth class labels of shape (num_samples,).
        num_classes (int): Number of classes.
        logger (Logger): Logger for debugging and info logging.

    Returns:
        dict: Dictionary containing MR, ER, RDR masks, RRV, and summary statistics per class.
        dict: Dictionary mapping unique activation patterns to sample indices.
    """
    RDR_AGREEMENT_THRESHOLD = config["rdr_agreement_threshold"]  # Use threshold from config
    
    logger.info(f"Activations shape: {activations.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Number of classes: {num_classes}")

    class_patterns = {c: defaultdict(int) for c in range(num_classes)}
    unique_patterns = {}
    pattern_index_map = {}
    class_samples = {c: [] for c in range(num_classes)}
    
    binary_activations = np.sign(activations)
    binary_activations[binary_activations == 0] = -1  # Convert zeros to -1
    
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
            }
            pattern_counter += 1

        unique_idx = pattern_index_map[pattern_tuple]
        unique_patterns[unique_idx]["original_activations"].append(activations[idx].tolist())
        unique_patterns[unique_idx]["samples"].append(idx)

        class_patterns[label][pattern_index_map[pattern_tuple]] += 1  
    
    results = {}

    for c in range(num_classes):
        if not class_patterns[c]:  
            continue  
    
        sorted_patterns = sorted(class_patterns[c].items(), key=lambda x: x[1], reverse=True)
        major_pattern_index, major_samples = sorted_patterns[0]  
        extra_regions = [{"activation_index": idx, "count": count} for idx, count in sorted_patterns[1:]]
        
        # Compute MRV (Mean Region Vector) - average of all activations for this class
        mrv = activations[class_samples[c]].mean(axis=0)
        
        # Compute RDR mask using majority agreement
        pattern_matrix = np.array([np.array(unique_patterns[idx]["activation_pattern"]) for idx, _ in sorted_patterns])
        
        # Feature must have the same sign for at least RDR_AGREEMENT_THRESHOLD % of samples
        sign_consistency = (np.mean(pattern_matrix == np.sign(np.sum(pattern_matrix, axis=0)), axis=0)) >= RDR_AGREEMENT_THRESHOLD
        classwise_rdr_mask = sign_consistency.astype(int)

        #Fixed: Compute RRV using indexing instead of multiplication
        rrv = np.zeros_like(mrv)  # Initialize with zeros
        rrv[classwise_rdr_mask == 1] = mrv[classwise_rdr_mask == 1]  # Select only masked values

        results[f"class_{c}"] = {
            "num_total_samples": sum(class_patterns[c].values()),
            "num_decision_regions": len(class_patterns[c]),
            "major_region": {
                "count": major_samples,
                "activation_index": major_pattern_index
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
