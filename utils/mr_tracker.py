import numpy as np
import json
import os

def compute_major_regions(activations, labels, num_classes):
    """
    Computes the Major Region (MR), Extra Regions (ER), and MRV (Major Region Vector) for each class.

    Args:
        activations (numpy.ndarray): Binary activation patterns of shape (num_samples, num_features).
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
        extra_regions = [{"activation_pattern": k, "samples": v} for k, v in sorted_patterns[1:]]

        # Compute Mean Vector for Major Region (MRV)
        major_indices = np.array(major_samples)
        mrv = activations[major_indices].mean(axis=0).tolist()

        results[f"class_{c}"] = {
            "major_region": {"activation_pattern": major_pattern, "samples": major_samples},
            "extra_regions": extra_regions,
            "mrv": mrv
        }

    return results

def save_major_regions(results, dataset_name, batch_size, output_dir="results/"):
    """
    Saves computed MR, ER, and MRV in a structured JSON file.

    Args:
        results (dict): MR & ER results computed from activations.
        dataset_name (str): Dataset name.
        batch_size (int): Batch size.
        output_dir (str): Directory where to save the results.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"major_regions_{dataset_name}_batch_{batch_size}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Major Region Data Saved: {output_path}")
