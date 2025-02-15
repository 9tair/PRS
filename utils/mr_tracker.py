import numpy as np
import torch
import json
import os

def compute_major_regions(activations, labels, num_classes):
    """
    Computes the Major Region (MR), Extra Regions (ER), and MRV (Major Region Vector) for each class.

    Args:
        activations (numpy.ndarray): Activation patterns of shape (num_samples, num_features).
        labels (numpy.ndarray): Ground-truth class labels of shape (num_samples,).
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary containing MR, ER, and MRV for each class.
    """
    class_patterns = {c: {} for c in range(num_classes)}  # Stores activation pattern counts per class

    # Convert activations to binary patterns
    binary_activations = np.sign(activations)  # Binarization (convert values to -1, 0, 1)
    binary_activations[binary_activations == 0] = -1  # Replace zeros with -1 for consistency

    # Track occurrences of activation patterns per class
    for idx, (pattern, label) in enumerate(zip(binary_activations, labels)):
        pattern_tuple = tuple(pattern)  # Convert pattern to hashable type
        if pattern_tuple not in class_patterns[label]:
            class_patterns[label][pattern_tuple] = []
        class_patterns[label][pattern_tuple].append(idx)

    # Compute MR and ER
    results = {}
    for c in range(num_classes):
        if not class_patterns[c]:
            continue

        # Find most frequent activation pattern (Major Region)
        sorted_patterns = sorted(class_patterns[c].items(), key=lambda x: len(x[1]), reverse=True)
        major_pattern, major_samples = sorted_patterns[0]

        # Extra regions = all other patterns
        extra_regions = [{"activation_pattern": list(k), "samples": v} for k, v in sorted_patterns[1:]]

        # Compute Mean Vector for Major Region (MRV)
        major_indices = np.array(major_samples)
        mrv = activations[major_indices].astype(np.float32).mean(axis=0).tolist()  # 🔹 Ensure float32

        results[f"class_{c}"] = {
            "major_region": {"activation_pattern": list(major_pattern), "samples": major_samples},
            "extra_regions": extra_regions,
            "mrv": mrv
        }

    return results


def convert_to_serializable(obj):
    """Convert NumPy/PyTorch objects into JSON-friendly Python types."""
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32).tolist()  # 🔹 Convert NumPy arrays to float32 lists
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().astype(np.float32).tolist()  # 🔹 Convert Tensors to float32 lists
    elif isinstance(obj, (np.float16, np.float32, np.float64)):  
        return float(obj)  # 🔹 Convert NumPy floats to Python float
    elif isinstance(obj, (np.int16, np.int32, np.int64)):  
        return int(obj)  # 🔹 Convert NumPy int to Python int
    elif isinstance(obj, tuple):  
        return list(obj)  # 🔹 Convert tuples to lists (JSON does not support tuples)
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}  # 🔹 Ensure dict keys are strings
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]  # 🔹 Convert list elements recursively
    else:
        return obj  # Return unchanged for Python-native types


def save_major_regions(major_regions, dataset_name, batch_size):
    """Save the major regions and associated samples in a JSON file."""
    save_path = f"results/major_regions_{dataset_name}_batch_{batch_size}.json"

    print(f"✅ Preparing Major Regions for saving: {save_path}")

    results = convert_to_serializable(major_regions)

    # 🔹 Debugging: Check data structure before saving
    print(f"🔹 Processed Data Structure (first class entry): {list(results.keys())[:1]}")
    print(f"🔹 Example data: {results[list(results.keys())[0]] if results else 'Empty'}")

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Major regions saved successfully to {save_path}")
