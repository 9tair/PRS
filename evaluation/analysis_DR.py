import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

# ---------------------- Load JSON Data Safely ----------------------
def load_json(filepath):
    """ Load JSON file safely, returning None if the file is missing """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

# ---------------------- Extract Model, Dataset, Batch Info from Folder ----------------------
def extract_info_from_folder(folder_path):
    """
    Extracts model name, dataset, and batch size from the provided folder path.
    Works with both standard paths and epoch-containing paths.
    
    Example folder structures:
    - /home/tair/project_root/models/saved/CNN-6_CIFAR10_batch_128/
    - /home/tair/project_root/models/saved/CNN-6_CIFAR10_batch_128/epoch_50/
    """
    # Check if we're in an epoch subdirectory
    folder_name = os.path.basename(folder_path)
    parent_folder = os.path.dirname(folder_path)
    
    # If we're in an epoch folder, use the parent directory for model info
    if folder_name.startswith('epoch_'):
        model_folder = os.path.basename(parent_folder)
        epoch_num = int(re.search(r'epoch_(\d+)', folder_name).group(1))
    else:
        model_folder = folder_name
        epoch_num = None
    
    # Extract model information from the model folder name
    match = re.search(r"(.+?)_(.+?)_batch_(\d+)", model_folder)
    if not match:
        raise ValueError(f"Invalid folder format: {model_folder}. Expected format: <MODEL>_<DATASET>_batch_<BATCH_SIZE>...")
    
    model_name, dataset_name, batch_size = match.groups()
    return model_name, dataset_name, int(batch_size), epoch_num

# ---------------------- Compute Classwise Intersection ----------------------
def compute_classwise_intersection(major_regions, unique_patterns):
    """Computes the intersection of activation coordinates across all decision regions for each class."""
    intersection_results = {}

    for class_label, class_data in major_regions.items():
        all_patterns = []
        for region in class_data["extra_regions"]:
            region_index = str(region["activation_index"])
            pattern = unique_patterns.get(region_index, {}).get("activation_pattern", [])
            if pattern:
                all_patterns.append(np.array(pattern))

        if not all_patterns:
            continue

        total_coordinates = len(all_patterns[0])
        intersection = np.all(np.array(all_patterns) == all_patterns[0], axis=0)
        common_coordinates_count = np.sum(intersection)

        intersection_results[class_label] = {
            "total_coordinates": total_coordinates,
            "common_coordinates_count": int(common_coordinates_count),
            "percentage_common": round(common_coordinates_count / total_coordinates, 4),
            "intersection_vector": intersection
        }

    return intersection_results

# ---------------------- Compute Intersection Across Classes ----------------------
def compute_intersection_across_classes(intersection_results):
    """Computes the intersection of intra-class common coordinates across different classes."""
    class_vectors = [data["intersection_vector"] for data in intersection_results.values()]
    
    if not class_vectors:
        return {"common_coordinates_count": 0}

    global_intersection = np.all(np.array(class_vectors), axis=0)
    global_common_count = np.sum(global_intersection)

    return {"common_coordinates_count": int(global_common_count)}

# ---------------------- Compare MR with Extra Regions ----------------------
def compare_mr_with_all_regions(major_regions, unique_patterns):
    """Compares the major region with all other regions in the same class."""
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

        class_comparisons.sort(key=lambda x: x["common_coordinates_count"], reverse=True)

        max_common = class_comparisons[0] if class_comparisons else None
        min_common = class_comparisons[-1] if class_comparisons else None

        comparison_results[class_label] = {
            "sorted_comparisons": class_comparisons,
            "max_common": max_common,
            "min_common": min_common
        }

    return comparison_results

# ---------------------- Visualization ----------------------
def plot_max_min_comparisons(comparison_results, intersection_results, global_intersection, output_path, model_name, dataset_name, batch_size, epoch_num=None):
    """
    Plots a bar chart of max, min common coordinates per class, intra-class intersection, and inter-class intersection.
    Added numerical values on top of each bar.
    """
    class_labels = list(comparison_results.keys())
    max_values = [comparison_results[c]["max_common"]["common_coordinates_count"] if comparison_results[c]["max_common"] else 0 for c in class_labels]
    min_values = [comparison_results[c]["min_common"]["common_coordinates_count"] if comparison_results[c]["min_common"] else 0 for c in class_labels]
    common_intersections = [intersection_results[c]["common_coordinates_count"] if c in intersection_results else 0 for c in class_labels]

    plt.figure(figsize=(14, 6))
    x = np.arange(len(class_labels))

    # Create bars with different positions
    max_bars = plt.bar(x - 0.3, max_values, width=0.25, color='green', label="Max Common Coordinates")
    min_bars = plt.bar(x, min_values, width=0.25, color='red', label="Min Common Coordinates")
    common_bars = plt.bar(x + 0.3, common_intersections, width=0.25, color='blue', label="Common Across All Regions")

    global_x = len(class_labels) + 0.5
    global_bars = plt.bar(global_x, global_intersection["common_coordinates_count"], width=0.4, color='black', label="Shared Across Classes")

    # Add the values on top of the bars
    def add_values_on_bars(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

    add_values_on_bars(max_bars)
    add_values_on_bars(min_bars)
    add_values_on_bars(common_bars)
    add_values_on_bars(global_bars)

    plt.xticks(list(x) + [global_x], class_labels + ["Global"], rotation=45)
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Common Coordinates")
    
    # Add epoch information to title if available
    title = f"Comparison of MR with Extra Regions - {model_name} {dataset_name}, Batch {batch_size}"
    if epoch_num is not None:
        title += f", Epoch {epoch_num}"
    
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add some top margin to accommodate the value labels
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()

# ---------------------- Main Execution ----------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize major regions vs. extra regions from a given folder.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing major_regions.json and unique_patterns.json")
    args = parser.parse_args()

    folder_path = args.folder

    # Extract model details from the folder name
    model_name, dataset_name, batch_size, epoch_num = extract_info_from_folder(folder_path)

    # Construct file paths
    major_regions_path = os.path.join(folder_path, "major_regions.json")
    
    # Look for either activation_patterns.json or unique_patterns.json
    activation_patterns_path = os.path.join(folder_path, "activation_patterns.json")
    if not os.path.exists(activation_patterns_path):
        activation_patterns_path = os.path.join(folder_path, "unique_patterns.json")

    # Load data
    major_regions = load_json(major_regions_path)
    unique_patterns = load_json(activation_patterns_path)

    if major_regions is None or unique_patterns is None:
        print("Error: Missing input data.")
        return

    # Perform Analysis
    comparison_mr_vs_all = compare_mr_with_all_regions(major_regions, unique_patterns)
    intersection_results = compute_classwise_intersection(major_regions, unique_patterns)
    global_intersection = compute_intersection_across_classes(intersection_results)

    # Generate visualization
    # Create appropriate filename based on epoch if applicable
    if epoch_num is not None:
        plot_filename = f"mr_comparison_plot_epoch_{epoch_num}.png"
    else:
        plot_filename = "mr_comparison_plot.png"
    
    plot_path = os.path.join(folder_path, plot_filename)
    plot_max_min_comparisons(
        comparison_mr_vs_all, 
        intersection_results, 
        global_intersection, 
        plot_path,
        model_name,
        dataset_name,
        batch_size,
        epoch_num
    )
    
    print(f"Analysis completed. Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()