import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import re

def extract_params_from_filepath(filepath):
    """Extracts model name, dataset, and batch size from the parent folder of the given filepath."""
    dir_path = os.path.dirname(filepath)  # Extract directory containing the file
    folder_name = os.path.basename(dir_path)  # Extract last folder in path

    # Example folder name: "VGG16_CIFAR10_batch_128_warmup_1_PRS"
    match = re.search(r"(.+?)_(.+?)_batch_(\d+)", folder_name)
    
    if not match:
        raise ValueError(f"Invalid directory format: {folder_name}. Expected format: <MODEL>_<DATASET>_batch_<BATCH_SIZE>_warmup_<N>_PRS")
    
    model_name, dataset_name, batch_size = match.groups()
    return model_name, dataset_name, int(batch_size), dir_path

def load_major_region_data(filepath):
    """Loads major region JSON data based on the provided file path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    
    print(f"🔹 Loading major region data from {filepath}...")
    with open(filepath, "r") as f:
        data = json.load(f)

    return data

def plot_mr_stacked_bar(filepath):
    """Generates a stacked bar plot for major and extra regions per class."""
    
    # Extract parameters from the parent folder
    model_name, dataset_name, batch_size, save_dir = extract_params_from_filepath(filepath)
    
    data = load_major_region_data(filepath)

    num_classes = 10  # Assuming CIFAR-10 / MNIST-style datasets
    expected_samples_per_class = 5000  # Expected count for balanced datasets

    class_sample_counts = np.zeros(num_classes)  # Total samples per class
    region_counts = [[] for _ in range(num_classes)]  # Store sorted region sizes
    count_one_regions = np.zeros(num_classes)  # Sum of all regions with count=1

    print("\n🔍 **Class-wise Sample Verification**:\n")
    for class_id in tqdm(range(num_classes), desc="Processing Classes"):
        class_key = f"class_{class_id}"
        if class_key not in data:
            continue

        major_region_samples = data[class_key]["major_region"]["count"]
        extra_region_samples = [region["count"] for region in data[class_key]["extra_regions"]]

        # Count and sum all regions with `count=1`
        count_one_regions[class_id] = sum(1 for count in extra_region_samples if count == 1)

        # Filter out count=1 from the extra regions
        filtered_extra_regions = [count for count in extra_region_samples if count > 1]

        # Sort remaining regions by size
        sorted_regions = [major_region_samples] + sorted(filtered_extra_regions, reverse=True)
        total_region_samples = sum(sorted_regions) + count_one_regions[class_id]

        class_sample_counts[class_id] = total_region_samples  # Should be 5000
        region_counts[class_id] = sorted_regions  # Store ordered region sizes

        # **Verification Message**
        if total_region_samples == expected_samples_per_class:
            print(f"Class {class_id} verification PASSED: {total_region_samples}/{expected_samples_per_class} samples")
        else:
            print(f"Class {class_id} verification FAILED: {total_region_samples}/{expected_samples_per_class} samples")
            print(f"   ➝ Major region: {major_region_samples}, Extra regions: {sum(filtered_extra_regions)}, Count=1 total: {count_one_regions[class_id]}")
    
    # 🔹 Stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(num_classes)  # Track bar height for stacking
    colors = plt.colormaps.get_cmap("tab20")

    for i in range(num_classes):
        for j, count in enumerate(region_counts[i]):
            ax.bar(i, count, bottom=bottom[i], color=colors(j % 20), edgecolor="white", width=0.8)
            bottom[i] += count  # Stack the regions

        # Add the aggregated "count=1" region bar (80% transparency)
        if count_one_regions[i] > 0:
            ax.bar(i, count_one_regions[i], bottom=bottom[i], color="red", alpha=0.2, edgecolor="black", width=0.8)
            bottom[i] += count_one_regions[i]  # Add to stack height

    # 🔹 Labels
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([f"{c}\n({int(class_sample_counts[c])})" for c in range(num_classes)])  # Add total sample count per class
    ax.set_title(f"Stacked Marginal and Extra Regions - {model_name} {dataset_name}, Batch {batch_size}")

    # 🔹 Save & show
    save_path = os.path.join(save_dir, "major_regions.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"\nPlot saved to: {save_path}")

if __name__ == "__main__":
    # **Command-Line Argument Parser**
    parser = argparse.ArgumentParser(description="Plot Marginal and Extra Regions using a JSON file.")

    parser.add_argument("--file", type=str, required=True, help="Full file path to the major region JSON file")

    args = parser.parse_args()

    # Call function with user-defined file
    plot_mr_stacked_bar(args.file)
