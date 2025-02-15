import os
import json
import numpy as np
import matplotlib.pyplot as plt

from config import config

def load_major_region_data(dataset_name, batch_size):
    """Load stored major region JSON data safely."""
    file_path = f"results/major_regions_{dataset_name}_batch_{batch_size}.json"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error in {file_path}: {e}")
        print(f"🛠️ Attempting to reload JSON with safe mode...")
        return load_json_safely(file_path)
    
    return data

def load_json_safely(file_path):
    """Try to read a potentially corrupted JSON file by trimming invalid content."""
    with open(file_path, "r", encoding="utf-8") as f:
        json_content = f.read()

    try:
        data = json.loads(json_content)
        return data
    except json.JSONDecodeError:
        print("⚠️ JSON is corrupted. Trying a manual fix...")
        
        # Attempt to fix missing end brackets
        while json_content and json_content[-1] not in ["}", "]"]:
            json_content = json_content[:-1]  # Trim last character

        try:
            data = json.loads(json_content)
            print("✅ Successfully recovered JSON after trimming invalid content.")
            return data
        except json.JSONDecodeError:
            raise ValueError(f"❌ Unable to recover JSON file: {file_path}")

def plot_mr_stacked_bar(dataset_name, batch_size):
    """Generate a stacked bar plot for marginal and extra regions per class."""
    print(f"🔹 Loading major region data for {dataset_name}, batch size {batch_size}...")
    data = load_major_region_data(dataset_name, batch_size)

    num_classes = 10  # Assuming CIFAR-10/MNIST-like dataset
    class_sample_counts = np.zeros(num_classes)  # Total samples per class
    region_counts = [[] for _ in range(num_classes)]  # Store sorted region sizes

    for class_id in range(num_classes):
        class_key = f"class_{class_id}"
        if class_key not in data:
            continue

        # Get region samples and sort them by size (largest first)
        major_region_samples = len(data[class_key]["major_region"]["samples"])
        extra_region_samples = [len(region["samples"]) for region in data[class_key]["extra_regions"]]
        
        sorted_regions = [major_region_samples] + sorted(extra_region_samples, reverse=True)  # Largest first

        class_sample_counts[class_id] = sum(sorted_regions)  # Total samples for the class
        region_counts[class_id] = sorted_regions  # Store ordered region sizes
    
    # Stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

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
    dataset = "CIFAR10"  # Change as needed
    batch_size = 128  # Change as needed
    plot_mr_stacked_bar(dataset, batch_size)
