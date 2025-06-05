import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm
import argparse
import re

def extract_params_from_filepath(filepath):
    """Extracts model name, dataset, and batch size from the filepath structure."""
    path_parts = filepath.split(os.sep)
    try:
        saved_idx = path_parts.index('saved')
    except ValueError:
        raise ValueError(f"Invalid path structure: 'saved' directory not found in {filepath}")
    if saved_idx + 1 >= len(path_parts):
        raise ValueError(f"Invalid path structure: no directory after 'saved' in {filepath}")
    model_dir_name = path_parts[saved_idx + 1]

    match = re.search(r"(.+?)_(.+?)_batch_(\d+)", model_dir_name)
    if not match:
        raise ValueError(
            f"Invalid directory format: {model_dir_name}. "
            "Expected format: <MODEL>_<DATASET>_batch_<BATCH_SIZE>"
        )
    model_name, dataset_name, batch_size = match.groups()
    save_dir = os.path.dirname(filepath)
    return model_name, dataset_name, int(batch_size), save_dir

def load_major_region_data(filepath):
    """Loads major region JSON data based on the provided file path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    print(f"🔹 Loading major region data from {filepath}...")
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def plot_mr_stacked_bar(filepath):
    """
    Generates a stacked bar plot for major and extra regions per class,
    coloring:
       • each 'count > 3' slice via tab20, but stepping through indices
         so that adjacent slices use very different colors,
         with alpha=0.8 for slight transparency,
       • all 'count == 1' in a semi-transparent teal (alpha=0.5),
       • 'count == 2' in semi-transparent orange (alpha=0.5),
       • 'count == 3' in semi-transparent green (alpha=0.5),
    and overlays a dashed black line with dot-markers connecting the
    major-region counts. The legend is removed from this main figure and
    drawn separately in its own figure. Y-axis labels are abbreviated as
    '0', '1k', '2k', ..., '5k'.
    """
    # ─── 1) Extract parameters and load data ─────────────────────────────────────
    model_name, dataset_name, batch_size, save_dir = extract_params_from_filepath(filepath)
    data = load_major_region_data(filepath)

    # ─── 2) Parse epoch info ─────────────────────────────────────────────────────
    epoch_match = re.search(r"epoch_(\d+)", filepath)
    epoch_info = f"Epoch {epoch_match.group(1)}" if epoch_match else "Final"

    num_classes = 10
    expected_samples_per_class = 5000  # e.g., CIFAR-10 is balanced

    # ─── 3) Prepare arrays for counts ────────────────────────────────────────────
    class_sample_counts  = np.zeros(num_classes)
    region_counts        = [[] for _ in range(num_classes)]  # [major] + sorted(>3)
    count_one_regions    = np.zeros(num_classes)  # sum of all extra regions where count == 1
    count_two_regions    = np.zeros(num_classes)  # sum of all extra regions where count == 2
    count_three_regions  = np.zeros(num_classes)  # sum of all extra regions where count == 3

    print("\n🔍 **Class-wise Sample Verification**:\n")
    for class_id in tqdm(range(num_classes), desc="Processing Classes"):
        class_key = f"class_{class_id}"
        if class_key not in data:
            continue

        major_region_samples = data[class_key]["major_region"]["count"]
        extra_counts = [region["count"] for region in data[class_key]["extra_regions"]]

        # Sum exact-size extras
        count_one_regions[class_id]   = sum(1 for c in extra_counts if c == 1)
        count_two_regions[class_id]   = sum(c for c in extra_counts if c == 2)
        count_three_regions[class_id] = sum(c for c in extra_counts if c == 3)

        # Keep only extra regions > 3
        filtered_large_regions = [c for c in extra_counts if c > 3]
        sorted_large = sorted(filtered_large_regions, reverse=True)

        # Build [major] + sorted(>3)
        region_counts[class_id] = [major_region_samples] + sorted_large

        total_samples = (
            major_region_samples
            + sum(sorted_large)
            + count_three_regions[class_id]
            + count_two_regions[class_id]
            + count_one_regions[class_id]
        )
        class_sample_counts[class_id] = total_samples

        if total_samples == expected_samples_per_class:
            print(f"Class {class_id} verification PASSED: {total_samples}/{expected_samples_per_class} samples")
        else:
            print(f"Class {class_id} verification FAILED: {total_samples}/{expected_samples_per_class} samples")
            print(
                f"   ➝ Major: {major_region_samples}, >3-sum: {sum(sorted_large)}, "
                f"3-sum: {count_three_regions[class_id]}, 2-sum: {count_two_regions[class_id]}, "
                f"1-sum: {count_one_regions[class_id]}"
            )

    # ─── 4) Draw the main stacked-bar + dashed-line plot (no legend) ────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(num_classes)
    cmap = plt.colormaps["tab20"]

    for i in range(num_classes):
        # a) Plot “major” + all “>3” bars, stepping through tab20 indices
        for j, csize in enumerate(region_counts[i]):
            # Step by 3 in the colormap index so adjacent slices get very different colors:
            color_index = (j * 3) % 20
            ax.bar(
                i,
                csize,
                bottom=bottom[i],
                color=cmap(color_index),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.3,
                width=0.8
            )
            bottom[i] += csize

        # b) Plot all “count == 3” slices as a single semi-transparent green bar
        if count_three_regions[i] > 0:
            ax.bar(
                i,
                count_three_regions[i],
                bottom=bottom[i],
                color="green",
                alpha=0.5,
                edgecolor="black",
                linewidth=0.3,
                width=0.8
            )
            bottom[i] += count_three_regions[i]

        # c) Plot all “count == 2” slices as a single semi-transparent orange bar
        if count_two_regions[i] > 0:
            ax.bar(
                i,
                count_two_regions[i],
                bottom=bottom[i],
                color="orange",
                alpha=0.5,
                edgecolor="black",
                linewidth=0.3,
                width=0.8
            )
            bottom[i] += count_two_regions[i]

        # d) Plot all “count == 1” slices as a single semi-transparent teal bar
        if count_one_regions[i] > 0:
            ax.bar(
                i,
                count_one_regions[i],
                bottom=bottom[i],
                color="#008080",  # teal
                alpha=0.5,        # 50% transparency
                edgecolor="black",
                linewidth=0.3,
                width=0.8
            )
            bottom[i] += count_one_regions[i]

    # e) Overlay the dashed line + dot-markers for major-region counts
    major_counts = [region_counts[i][0] for i in range(num_classes)]
    ax.plot(
        list(range(num_classes)),
        major_counts,
        marker='o',
        linestyle='--',
        color='black',
        linewidth=1.0,
        markersize=5
    )

    # ─── 5) Tidy up the axes (no title, no y-label, no outer box) ──────────────
    ax.set_title("")          # remove title
    ax.set_ylabel("")         # remove y-axis label

    # Only show bottom & left spines, hide top & right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    # Replace x-tick labels with just “0 … 9” (no “(5000)” suffix)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([str(i) for i in range(num_classes)])

    # Abbreviate y-axis ticks as '0', '1k', '2k', ..., '5k'
    def thousands_formatter(x, pos):
        if x == 0:
            return "0"
        else:
            return f"{int(x/1000)}k"
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))

    # Turn off any built-in grid
    ax.grid(False)

    # ─── 6) Remove left padding (make bars flush with the y-axis) ──────────────
    ax.set_xlim(-0.5, num_classes - 0.5)

    # ─── 7) Show and save the main plot ─────────────────────────────────────────
    plt.tight_layout(pad=2)
    main_plot_filename = f"major_regions_{epoch_info.replace(' ', '_').lower()}.png"
    main_save_path = os.path.join(save_dir, main_plot_filename)
    plt.savefig(main_save_path, bbox_inches="tight")
    plt.show()
    print(f"\nMain plot saved to: {main_save_path}")

    # ─── 8) Create a separate legend-only figure ──────────────────────────────────
    legend_handles = [
        Line2D([], [], marker='o', linestyle='--', color='black', linewidth=1.0,
               markersize=5, markerfacecolor='black', label="Major-region count"),
        Patch(facecolor='green', edgecolor='black', label="3-sample regions", alpha=0.5),
        Patch(facecolor='orange', edgecolor='black', label="2-sample regions", alpha=0.5),
        Patch(facecolor='#008080', edgecolor='black', label="1-sample regions", alpha=0.5)
    ]

    fig2, ax2 = plt.subplots(figsize=(4, 2))
    ax2.axis('off')  # no axes lines or ticks—just show the legend
    ax2.legend(handles=legend_handles, loc='center', frameon=False)
    plt.tight_layout()

    legend_filename = f"legend_only_{epoch_info.replace(' ', '_').lower()}.png"
    legend_save_path = os.path.join(save_dir, legend_filename)
    plt.savefig(legend_save_path, bbox_inches="tight")
    plt.show()
    print(f"Legend-only plot saved to: {legend_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Marginal and Extra Regions using a JSON file.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Full file path to the major region JSON file"
    )
    args = parser.parse_args()
    plot_mr_stacked_bar(args.file)