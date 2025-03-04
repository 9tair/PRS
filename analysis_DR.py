import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Load JSON Data Safely ----------------------
def load_json(filepath):
    """ Load JSON file safely, returning None if the file is missing """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

# ---------------------- Compute Common Coordinates Across All Regions in a Class ----------------------
def compute_classwise_intersection(major_regions, unique_patterns):
    """
    Computes the intersection of activation coordinates across all decision regions for each class.

    Returns:
        dict: Contains intersection size for each class.
    """
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
            "intersection_vector": intersection  # Store the actual intersection vector
        }

    return intersection_results

# ---------------------- Compute Intersection Across Classes ----------------------
def compute_intersection_across_classes(intersection_results):
    """
    Computes the intersection of intra-class common coordinates across different classes.

    Returns:
        dict: Contains a single entry with the total shared coordinates.
    """
    class_vectors = [data["intersection_vector"] for data in intersection_results.values()]
    
    if not class_vectors:
        return {"common_coordinates_count": 0}

    # Compute intersection across all classes
    global_intersection = np.all(np.array(class_vectors), axis=0)
    global_common_count = np.sum(global_intersection)

    return {"common_coordinates_count": int(global_common_count)}

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

        # Sort by decreasing order of common coordinates
        class_comparisons.sort(key=lambda x: x["common_coordinates_count"], reverse=True)

        # Get max and min comparisons
        max_common = class_comparisons[0] if class_comparisons else None
        min_common = class_comparisons[-1] if class_comparisons else None

        comparison_results[class_label] = {
            "sorted_comparisons": class_comparisons,
            "max_common": max_common,
            "min_common": min_common
        }

    return comparison_results

# ---------------------- Visualization: Max, Min & Class Intersection ----------------------
def plot_max_min_comparisons(comparison_results, intersection_results, global_intersection, output_path):
    """
    Plots a bar chart of max, min common coordinates per class, the number of coordinates
    that are the same across all regions in a class, and a final bar for the global intersection across classes.
    
    Args:
        comparison_results (dict): Dictionary of MR comparisons.
        intersection_results (dict): Dictionary of common coordinate intersections per class.
        global_intersection (dict): Dictionary with a single value for shared coordinates across classes.
        output_path (str): Path to save the plot.
    """
    class_labels = list(comparison_results.keys())
    max_values = [comparison_results[c]["max_common"]["common_coordinates_count"] if comparison_results[c]["max_common"] else 0 for c in class_labels]
    min_values = [comparison_results[c]["min_common"]["common_coordinates_count"] if comparison_results[c]["min_common"] else 0 for c in class_labels]
    common_intersections = [intersection_results[c]["common_coordinates_count"] if c in intersection_results else 0 for c in class_labels]

    plt.figure(figsize=(14, 6))
    x = np.arange(len(class_labels))

    # Create bars
    bars_max = plt.bar(x - 0.3, max_values, width=0.25, color='green', label="Max Common Coordinates")
    bars_min = plt.bar(x, min_values, width=0.25, color='red', label="Min Common Coordinates")
    bars_common = plt.bar(x + 0.3, common_intersections, width=0.25, color='blue', label="Common Across All Regions")

    # Final bar for inter-class intersection
    global_x = len(class_labels) + 0.5
    global_bar = plt.bar(global_x, global_intersection["common_coordinates_count"], width=0.4, color='black', label="Shared Across Classes")

    # Add exact numerical values on top of bars
    for bar in bars_max:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=10, fontweight="bold", color="black")

    for bar in bars_min:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=10, fontweight="bold", color="black")

    for bar in bars_common:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=10, fontweight="bold", color="black")

    plt.text(global_x, global_bar[0].get_height(), f"{global_bar[0].get_height():.0f}", ha='center', va='bottom', fontsize=10, fontweight="bold", color="black")

    plt.xticks(list(x) + [global_x], class_labels + ["Global"], rotation=45)
    plt.xlabel("Class Labels", fontsize=12)
    plt.ylabel("Number of Common Coordinates", fontsize=12)
    plt.title("Comparison of MR with Extra Regions (Max vs. Min), Intra-Class Intersection, and Global Intersection", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and Show Plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()

# ---------------------- Main Execution ----------------------
def main():
    dataset_name = "CIFAR10"  
    batch_size = 128
    model_name = "VGG16"

    # Paths to JSON files
    major_regions_path = f"results/major_regions_{model_name}_{dataset_name}_batch_{batch_size}.json"
    activation_patterns_path = f"results/activation_patterns_{model_name}_{dataset_name}_batch_{batch_size}.json"

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
    plot_path = f"results/mr_comparison_plot_{model_name}_{dataset_name}_batch_{batch_size}.png"
    plot_max_min_comparisons(comparison_mr_vs_all, intersection_results, global_intersection, plot_path)

if __name__ == "__main__":
    main()
