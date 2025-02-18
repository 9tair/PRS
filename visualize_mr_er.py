import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import config

def load_major_region_data(dataset_name, batch_size):
    """Efficiently load compressed major region JSON data."""
    file_path = f"results/major_regions_{dataset_name}_batch_{batch_size}.json"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File {file_path} not found.")

    print(f"🔹 Loading major region data for {dataset_name}, batch size {batch_size}...")
    with open(file_path, "r") as f:
        data = json.load(f)

    return data

def plot_mr_stacked_bar(dataset_name, batch_size):
    """Generate a stacked bar plot for marginal and extra regions per class."""
    data = load_major_region_data(dataset_name, batch_size)

    num_classes = 10  # Assuming CIFAR-10 / MNIST-style datasets
    total_samples_per_class = 5000  # ✅ Each class should have 5000 training samples

    class_sample_counts = np.zeros(num_classes)  # Track total samples per class
    region_counts = [[] for _ in range(num_classes)]  # Store sorted region sizes

    print("🔄 Processing data for visualization...")
    for class_id in tqdm(range(num_classes), desc="Processing Classes"):
        class_key = f"class_{class_id}"
        if class_key not in data:
            continue

        # Extract major region + extra regions
        major_region_samples = data[class_key]["major_region"]["count"]
        extra_region_samples = [region["count"] for region in data[class_key]["extra_regions"]]
        
        sorted_regions = [major_region_samples] + sorted(extra_region_samples, reverse=True)  # Largest first
        total_region_samples = sum(sorted_regions)

        # Validate that the sum equals 5000
        if total_region_samples != total_samples_per_class:
            print(f"⚠️ Warning: Class {class_id} has {total_region_samples} samples instead of {total_samples_per_class}")

        class_sample_counts[class_id] = total_region_samples  # ✅ Should be 5000
        region_counts[class_id] = sorted_regions  # Store ordered region sizes
    
    # Stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(num_classes)  # Track bar height for stacking
    for i in range(num_classes):
        for j, count in enumerate(region_counts[i]):
            ax.bar(i, count, bottom=bottom[i], color=plt.cm.tab20(j / 20), edgecolor="white", width=0.8)
            bottom[i] += count  # Stack the regions

    # Labels
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([f"{c}\n({len(region_counts[c])} regions)" for c in range(num_classes)])  # Add region count
    ax.set_title(f"Stacked Marginal and Extra Regions - {dataset_name}, Batch {batch_size}")

    # Save & show
    save_path = f"results/mr_visualization_{dataset_name}_batch_{batch_size}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"✅ Plot saved to {save_path}")

if __name__ == "__main__":
    dataset = "CIFAR10"
    batch_size = 128  # Change as needed
    plot_mr_stacked_bar(dataset, batch_size)
