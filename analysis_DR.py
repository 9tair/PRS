import os
import json
import numpy as np

# ---------------------- Load JSON Data Safely ----------------------
def load_json(filepath):
    """ Load JSON file safely, returning None if the file is missing """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

# ---------------------- Compare MR with All Regions in the Same Class ----------------------
def compare_mr_with_all_regions(major_regions, unique_patterns):
    """ Compare MR with all other regions in the same class. """
    if major_regions is None or unique_patterns is None:
        return {}

    comparison_results = {}

    for class_label, class_data in major_regions.items():
        if "major_region" not in class_data or "extra_regions" not in class_data:
            continue
        
        mr_index = class_data["major_region"]["activation_index"]
        mr_vector = np.array(unique_patterns.get(str(mr_index), {}).get("activation_pattern", []))
        total_coordinates = len(mr_vector)

        class_comparisons = []
        for extra_region in class_data["extra_regions"]:
            extra_index = extra_region["activation_index"]
            extra_vector = np.array(unique_patterns.get(str(extra_index), {}).get("activation_pattern", []))

            if len(extra_vector) == 0 or len(mr_vector) == 0:
                continue

            common_coordinates = np.sum(mr_vector == extra_vector)
            class_comparisons.append({
                "region_index": extra_index,
                "common_coordinates_count": int(common_coordinates),
                "total_coordinates": total_coordinates,
                "percentage_common": round(common_coordinates / total_coordinates, 4)
            })

        comparison_results[class_label] = class_comparisons

    return comparison_results

# ---------------------- Compare MRs Across All Classes ----------------------
def compare_mr_across_classes(major_regions, unique_patterns):
    """ Compare MRs of different classes and find shared coordinates. """
    if major_regions is None or unique_patterns is None:
        return {}

    mr_vectors = {}
    for class_label, class_data in major_regions.items():
        if "major_region" in class_data:
            mr_index = class_data["major_region"]["activation_index"]
            mr_vectors[class_label] = np.array(unique_patterns.get(str(mr_index), {}).get("activation_pattern", []))

    class_labels = list(mr_vectors.keys())
    total_coordinates = len(next(iter(mr_vectors.values()), []))

    if total_coordinates == 0:
        return {}

    comparison_results = {}
    for i in range(len(class_labels)):
        for j in range(i + 1, len(class_labels)):
            class_a, class_b = class_labels[i], class_labels[j]
            common_coordinates = np.sum(mr_vectors[class_a] == mr_vectors[class_b])

            if class_a not in comparison_results:
                comparison_results[class_a] = {}
            if class_b not in comparison_results:
                comparison_results[class_b] = {}

            comparison_results[class_a][class_b] = {
                "common_coordinates_count": int(common_coordinates),
                "total_coordinates": total_coordinates,
                "percentage_common": round(common_coordinates / total_coordinates, 4)
            }

            comparison_results[class_b][class_a] = comparison_results[class_a][class_b]

    return comparison_results

# ---------------------- Compute Variance Per Coordinate Per Class ----------------------
def compute_variance_per_coordinate(major_regions, unique_patterns):
    """ Compute variance per coordinate per class. """
    if major_regions is None or unique_patterns is None:
        return {}

    variance_results = {}

    for class_label, class_data in major_regions.items():
        region_vectors = []
        for region in class_data["extra_regions"]:
            region_index = region["activation_index"]
            region_vectors.append(np.array(unique_patterns.get(str(region_index), {}).get("activation_pattern", [])))

        if len(region_vectors) > 0:
            region_vectors = np.array(region_vectors)
            variance_per_class = np.var(region_vectors, axis=0).tolist()
            variance_results[class_label] = variance_per_class

    return variance_results

# ---------------------- Main Execution ----------------------
def main():
    dataset_name = "CIFAR10"  
    batch_size = 128
    model_name = "CNN-6"

    # Paths to JSON files
    major_regions_path = f"results/weak_baseline/major_regions_{model_name}_{dataset_name}_batch_{batch_size}.json"
    activation_patterns_path = f"results/weak_baseline/activation_patterns_{model_name}_{dataset_name}_batch_{batch_size}.json"

    # Load data
    major_regions = load_json(major_regions_path)
    unique_patterns = load_json(activation_patterns_path)

    if major_regions is None or unique_patterns is None:
        print("Error: Missing input data. Please check if the JSON files are generated.")
        return

    # Perform Analysis
    analysis_results = {
        "comparison_mr_vs_all_regions": compare_mr_with_all_regions(major_regions, unique_patterns),
        "comparison_mr_across_classes": compare_mr_across_classes(major_regions, unique_patterns),
        "variance_per_coordinate_per_class": compute_variance_per_coordinate(major_regions, unique_patterns),
    }

    # Save results in JSON format
    output_path = f"results/analysis_results_{model_name}_{dataset_name}_batch_{batch_size}.json"
    with open(output_path, "w") as f:
        json.dump(analysis_results, f, indent=4)

    print(f"Analysis saved at: {output_path}")

if __name__ == "__main__":
    main()
