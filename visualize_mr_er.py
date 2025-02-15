import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from config import config

def load_major_regions(dataset_name, batch_size):
    """Load saved major region (MR) and extra region (ER) data from JSON."""
    file_path = f"results/major_regions_{dataset_name}_batch_{batch_size}.json"

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    with open(file_path, "r") as f:
        data = json.load(f)

    return data

def process_mr_data(mr_data):
    """
    Extracts the number of samples per decision region for each class.
    Returns structured data for stacked bar plotting.
    """
    class_ids = []
    region_samples = []  # Each inner list is a class containing sample counts per region
    max_regions = 0

    for class_name, class_data in mr_data.items():
        class_id = int(class_name.split("_")[-1])  # Extract class index
        regions = []

        # Major Region samples
        major_samples = len(class_data["major_region"]["samples"])
        regions.append(major_samples)

        # Extra Region samples
        for region in class_data["extra_regions"]:
            regions.append(len(region["samples"]))

        class_ids.append(class_id)
        region_samples.append(regions)
        max_regions = max(max_regions, len(regions))

    return class_ids, region_samples, max_regions

def plot_stacked_mr(dataset_name, batch_size):
    """Generate a stacked bar chart of decision regions per class with better scaling."""
    print(f"🔹 Loading major region data for {dataset_name}, batch size {batch_size}...")
    mr_data = load_major_regions(dataset_name, batch_size)

    if mr_data is None:
        return

    class_ids, region_samples, max_regions = process_mr_data(mr_data)

    # Sort classes for better visualization
    sorted_indices = np.argsort(class_ids)
    class_ids = np.array(class_ids)[sorted_indices]
    region_samples = [region_samples[i] for i in sorted_indices]

    # **🔥 Fix: Limit the figure size to prevent huge images**
    num_classes = len(class_ids)
    fig_height = min(8 + (num_classes * 0.3), 20)  # Adaptive figure height
    fig_width = max(10, num_classes * 0.5)  # Adaptive figure width

    plt.figure(figsize=(fig_width, fig_height))
    bottom = np.zeros(len(class_ids))

    # **🎨 Fix: Use a colormap with limited distinct colors**
    cmap = cm.get_cmap("tab20", max_regions)  # Use up to 20 colors
    colors = [mcolors.to_rgba(cmap(i)) for i in range(max_regions)]

    # **📏 Fix: Handle too many regions**
    region_cutoff = 15  # If more than this, group small ones
    other_region_count = np.zeros(len(class_ids))

    for region_idx in range(max_regions):
        region_counts = [regions[region_idx] if region_idx < len(regions) else 0 for regions in region_samples]

        if region_idx < region_cutoff:
            plt.bar(class_ids, region_counts, bottom=bottom, color=colors[region_idx], label=f"Region {region_idx+1}")
        else:
            other_region_count += np.array(region_counts)

        bottom += np.array(region_counts)

    # Add "Other Regions" if many small ones exist
    if max_regions > region_cutoff:
        plt.bar(class_ids, other_region_count, bottom=bottom, color="gray", label="Other Regions")

    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(class_ids, [f"Class {c}" for c in class_ids])
    plt.title(f"Decision Regions per Class - {dataset_name} (Batch {batch_size})")
    plt.legend(title="Decision Regions", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # **🔥 Fix: Handle large saving issues**
    save_path = f"results/mr_region_visualization_{dataset_name}_batch_{batch_size}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)  # Limit DPI to avoid oversized images
    plt.show()

    print(f"✅ Visualization saved to: {save_path}")

if __name__ == "__main__":
    for dataset in config["datasets"]:
        for batch_size in config["batch_sizes"]:
            plot_stacked_mr(dataset, batch_size)
