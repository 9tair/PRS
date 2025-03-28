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


# ---------------------- Compare One Class Against Others ----------------------
def compare_class_against_others(target_class, major_regions, unique_patterns):
    """
    Compares each activation pattern from the target class against every individual 
    region from every other class.
    
    Additionally, compares only the Major Region (MR) of the target class against all regions
    from other classes, and against the MR of other classes.
    
    Also computes the intersection of all regions from target class with all regions from each other class.
    Returns detailed statistics about these comparisons.
    """
    target_data = major_regions.get(target_class)
    if not target_data:
        return {}

    # Check if target class has a major region
    if "major_region" not in target_data or "activation_index" not in target_data["major_region"]:
        return {"error": "Target class does not have a major region"}
    
    # Get major region index and pattern
    mr_index = target_data["major_region"]["activation_index"]
    mr_pattern = unique_patterns.get(str(mr_index), {}).get("activation_pattern", [])
    if not mr_pattern:
        return {"error": "Major region pattern not found for target class"}
    
    mr_vector = np.array(mr_pattern)
    
    # Get all regions from target class (MR + extras)
    target_indices = [mr_index]  # Start with the MR
    
    for region in target_data.get("extra_regions", []):
        if "activation_index" in region:
            target_indices.append(region["activation_index"])
    
    # Get all target vectors
    target_vectors = []
    for idx in target_indices:
        pattern = unique_patterns.get(str(idx), {}).get("activation_pattern", [])
        if pattern:
            target_vectors.append({
                "index": idx,
                "vector": np.array(pattern)
            })
    
    if not target_vectors:
        return {"error": "No valid vectors found for target class"}
    
    total_coordinates = len(target_vectors[0]["vector"])
    
    # Compute intersection of all target class regions
    # First check if all vectors have the same length
    if not all(len(tv["vector"]) == total_coordinates for tv in target_vectors):
        return {"error": "Not all target vectors have the same length"}
    
    # Find coordinates that are the same across all target regions
    target_arrays = np.array([tv["vector"] for tv in target_vectors])
    if len(target_arrays) > 1:
        # Check if all vectors have the same value at each position
        target_intersection = np.all(target_arrays == target_arrays[0], axis=0)
        # Count how many coordinates have the same value across all vectors
        target_common_count = np.sum(target_intersection)
    else:
        # If there's only one vector, all coordinates match by definition
        target_intersection = np.ones(total_coordinates, dtype=bool)
        target_common_count = total_coordinates
    
    # Prepare results structure
    results = {
        "region_comparisons": [],
        "summary": {
            "max_overlap": 0,
            "min_overlap": total_coordinates,
            "avg_overlap": 0,
            "total_comparisons": 0
        },
        "class_summaries": {},
        "class_intersections": {},  # Intersection of all regions
        "mr_comparisons": {},  # MR vs other class regions
        "mr_to_mr_overlap": {}  # NEW: MR vs MR of other classes
    }
    
    # Compare each target region with each region from other classes
    all_overlaps = []
    
    for target_vec_info in target_vectors:
        target_idx = target_vec_info["index"]
        target_vec = target_vec_info["vector"]
        
        region_result = {
            "target_region_index": target_idx,
            "comparisons_with_other_classes": {}
        }
        
        # For each other class
        for other_class, other_data in major_regions.items():
            if other_class == target_class:
                continue
                
            # Initialize class summary if needed
            if other_class not in results["class_summaries"]:
                results["class_summaries"][other_class] = {
                    "max_overlap": 0,
                    "min_overlap": total_coordinates,
                    "avg_overlap": 0,
                    "total_comparisons": 0
                }
                
            # Initialize MR comparisons if needed
            if other_class not in results["mr_comparisons"]:
                results["mr_comparisons"][other_class] = {
                    "max_overlap": 0,
                    "min_overlap": total_coordinates,
                    "avg_overlap": 0,
                    "total_comparisons": 0
                }
                
            # Get other class's major region if it exists
            other_mr_index = None
            other_mr_vector = None
            if "major_region" in other_data and "activation_index" in other_data["major_region"]:
                other_mr_index = other_data["major_region"]["activation_index"]
                other_mr_pattern = unique_patterns.get(str(other_mr_index), {}).get("activation_pattern", [])
                if other_mr_pattern and len(other_mr_pattern) == total_coordinates:
                    other_mr_vector = np.array(other_mr_pattern)
                    
                    # If this is the target MR and we have other MR, compute MR-to-MR overlap
                    if target_idx == mr_index:
                        overlap_count = np.sum(target_vec == other_mr_vector)
                        overlap_percent = round(overlap_count / total_coordinates, 4)
                        
                        results["mr_to_mr_overlap"][other_class] = {
                            "overlap_count": int(overlap_count),
                            "overlap_percent": overlap_percent,
                            "total_coordinates": total_coordinates
                        }
                
            # Get all regions from this other class
            other_indices = []
            if other_mr_index is not None:
                other_indices.append(other_mr_index)
                
            for region in other_data.get("extra_regions", []):
                if "activation_index" in region:
                    other_indices.append(region["activation_index"])
            
            class_comparisons = []
            mr_comparisons = []  # Store comparisons specifically for the MR
            
            # Collect all vectors for this other class for intersection calculation
            other_vectors = []
            
            # Compare target vector with each region in this class
            for other_idx in other_indices:
                other_pattern = unique_patterns.get(str(other_idx), {}).get("activation_pattern", [])
                if not other_pattern or len(other_pattern) != len(target_vec):
                    continue
                    
                other_vec = np.array(other_pattern)
                other_vectors.append(other_vec)
                
                overlap_count = np.sum(target_vec == other_vec)
                overlap_percent = round(overlap_count / total_coordinates, 4)
                
                comparison = {
                    "other_region_index": other_idx,
                    "overlap_count": int(overlap_count),
                    "overlap_percent": overlap_percent
                }
                
                class_comparisons.append(comparison)
                all_overlaps.append(overlap_count)
                
                # For the major region specifically
                if target_idx == mr_index:
                    mr_comparisons.append(comparison)
                
                # Update class summary
                results["class_summaries"][other_class]["max_overlap"] = max(
                    results["class_summaries"][other_class]["max_overlap"], 
                    overlap_count
                )
                results["class_summaries"][other_class]["min_overlap"] = min(
                    results["class_summaries"][other_class]["min_overlap"], 
                    overlap_count
                )
                results["class_summaries"][other_class]["total_comparisons"] += 1
            
            # Calculate the intersection between all target regions and all regions of this other class
            if other_vectors:
                other_arrays = np.array(other_vectors)
                if len(other_arrays) > 1:
                    # Find coordinates that are the same across all regions in other class
                    other_intersection = np.all(other_arrays == other_arrays[0], axis=0)
                    other_common_count = np.sum(other_intersection)
                    
                    # Now compare target intersection with other class intersection
                    # For a coordinate to be in the final intersection:
                    # 1. It must be in the intersection of all target class regions (have the same value)
                    # 2. It must be in the intersection of all other class regions (have the same value)
                    # 3. The value in the target intersection must match the value in the other intersection
                    
                    # Points where both classes have internal consensus (all regions agree within their class)
                    both_consistent = target_intersection & other_intersection
                    
                    # Among those points, find where the values actually match between classes
                    # First get the consensus value for each class (we know all values are the same within each class at these positions)
                    target_consensus_values = target_arrays[0][target_intersection]
                    other_consensus_values = other_arrays[0][other_intersection]
                    
                    # Since the vectors might have different lengths after filtering, we need to be careful with this comparison
                    # Let's create full boolean arrays first and then compare
                    target_values = np.zeros(total_coordinates, dtype=int)
                    other_values = np.zeros(total_coordinates, dtype=int)
                    
                    target_values[target_intersection] = target_consensus_values
                    other_values[other_intersection] = other_consensus_values
                    
                    # Now find positions where values match and both classes have internal consensus
                    full_intersection = both_consistent & (target_values == other_values)
                    intersection_count = np.sum(full_intersection)
                else:
                    # If other class has only one region, compare it with target intersection
                    other_region = other_arrays[0]
                    target_consensus_values = target_arrays[0][target_intersection]
                    other_values = other_region[target_intersection]
                    value_match = target_consensus_values == other_values
                    intersection_count = np.sum(value_match)
                
                results["class_intersections"][other_class] = {
                    "intersection_count": int(intersection_count),
                    "percentage": round(intersection_count / total_coordinates, 4)
                }
            
            # Process the Major Region comparisons
            if mr_comparisons:
                mr_overlaps = [comp["overlap_count"] for comp in mr_comparisons]
                results["mr_comparisons"][other_class] = {
                    "max_overlap": int(max(mr_overlaps)),
                    "min_overlap": int(min(mr_overlaps)),
                    "avg_overlap": round(sum(mr_overlaps) / len(mr_overlaps), 2),
                    "total_comparisons": len(mr_overlaps)
                }
            
            # Sort comparisons by overlap (highest first)
            class_comparisons.sort(key=lambda x: x["overlap_count"], reverse=True)
            
            # Calculate average for this class
            if class_comparisons:
                avg_overlap = sum(comp["overlap_count"] for comp in class_comparisons) / len(class_comparisons)
                results["class_summaries"][other_class]["avg_overlap"] = round(avg_overlap, 2)
            
            # Add to region result
            region_result["comparisons_with_other_classes"][other_class] = class_comparisons
        
        # Add this region's comparisons to overall results
        results["region_comparisons"].append(region_result)
    
    # Calculate overall summary
    if all_overlaps:
        results["summary"]["max_overlap"] = int(max(all_overlaps))
        results["summary"]["min_overlap"] = int(min(all_overlaps))
        results["summary"]["avg_overlap"] = round(sum(all_overlaps) / len(all_overlaps), 2)
        results["summary"]["total_comparisons"] = len(all_overlaps)
    
    return results

# ---------------------- Plot Class vs Others ----------------------
def plot_class_vs_others(target_class, comparison_results, output_folder):
    """
    Plots overlap of target class regions with each other class.
    Shows two sets of metrics:
    1. All regions: max, min, average overlap, and intersection
    2. Major Region only: max, min, average overlap, and MR-to-MR overlap
    
    All bars include numeric labels.
    """
    if "error" in comparison_results:
        print(f"Error plotting class {target_class}: {comparison_results['error']}")
        return
        
    class_summaries = comparison_results.get("class_summaries", {})
    class_intersections = comparison_results.get("class_intersections", {})
    mr_comparisons = comparison_results.get("mr_comparisons", {})
    mr_to_mr_overlap = comparison_results.get("mr_to_mr_overlap", {})
    
    if not class_summaries:
        print(f"No valid comparison data for class {target_class}")
        return
        
    other_classes = list(class_summaries.keys())
    
    # Values for all regions
    avg_overlaps = [class_summaries[c]["avg_overlap"] for c in other_classes]
    max_overlaps = [class_summaries[c]["max_overlap"] for c in other_classes]
    min_overlaps = [class_summaries[c]["min_overlap"] for c in other_classes]
    intersection_overlaps = [class_intersections.get(c, {}).get("intersection_count", 0) for c in other_classes]
    
    # Values for MR only
    mr_avg_overlaps = [mr_comparisons.get(c, {}).get("avg_overlap", 0) for c in other_classes]
    mr_max_overlaps = [mr_comparisons.get(c, {}).get("max_overlap", 0) for c in other_classes]
    mr_min_overlaps = [mr_comparisons.get(c, {}).get("min_overlap", 0) for c in other_classes]
    mr_to_mr_values = [mr_to_mr_overlap.get(c, {}).get("overlap_count", 0) for c in other_classes]
    
    # Create the plot with two rows - one for all regions, one for MR
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    x = np.arange(len(other_classes))
    width = 0.2  # Width of bars
    
    # Plot data for all regions in the first subplot
    avg_bars1 = ax1.bar(x - 1.5*width, avg_overlaps, width, color='blue', label='Average Overlap')
    max_bars1 = ax1.bar(x - 0.5*width, max_overlaps, width, color='green', label='Max Overlap')
    min_bars1 = ax1.bar(x + 0.5*width, min_overlaps, width, color='red', label='Min Overlap')
    int_bars1 = ax1.bar(x + 1.5*width, intersection_overlaps, width, color='purple', label='All Regions Intersection')
    
    # Plot data for MR only in the second subplot
    avg_bars2 = ax2.bar(x - 1.5*width, mr_avg_overlaps, width, color='blue', label='MR Avg Overlap')
    max_bars2 = ax2.bar(x - 0.5*width, mr_max_overlaps, width, color='green', label='MR Max Overlap')
    min_bars2 = ax2.bar(x + 0.5*width, mr_min_overlaps, width, color='red', label='MR Min Overlap')
    mr_bars2 = ax2.bar(x + 1.5*width, mr_to_mr_values, width, color='orange', label='MR-to-MR Overlap')
    
    # Add numeric labels on top of each bar
    def add_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                ax.text(bar.get_x() + bar.get_width()/2, height + 3,
                      f'{int(height) if float(height).is_integer() else round(float(height), 1)}',
                      ha='center', va='bottom', fontsize=8)
    
    # Add labels to all bars
    add_labels(ax1, avg_bars1)
    add_labels(ax1, max_bars1)
    add_labels(ax1, min_bars1)
    add_labels(ax1, int_bars1)
    
    add_labels(ax2, avg_bars2)
    add_labels(ax2, max_bars2)
    add_labels(ax2, min_bars2)
    add_labels(ax2, mr_bars2)
    
    # Set titles and labels
    ax1.set_title(f"All Regions of Class {target_class} vs Other Classes")
    ax2.set_title(f"Only Major Region of Class {target_class} vs Other Classes")
    
    ax2.set_xlabel("Other Classes")
    ax1.set_ylabel("Coordinates Overlap")
    ax2.set_ylabel("Coordinates Overlap")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(other_classes, rotation=45)
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Add summary text
    overall_summary = comparison_results.get("summary", {})
    if overall_summary:
        summary_text = (
            f"Overall: Avg={overall_summary.get('avg_overlap', 'N/A')}, "
            f"Max={overall_summary.get('max_overlap', 'N/A')}, "
            f"Min={overall_summary.get('min_overlap', 'N/A')}, "
            f"Total Comparisons={overall_summary.get('total_comparisons', 'N/A')}"
        )
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=9)
    
    # Adjust plot layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    
    # Save the figure
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"class_{target_class}_vs_others.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    intersection_overlaps = [class_intersections.get(c, {}).get("intersection_count", 0) for c in other_classes]
    
    # Values for MR only
    mr_avg_overlaps = [mr_comparisons.get(c, {}).get("avg_overlap", 0) for c in other_classes]
    mr_max_overlaps = [mr_comparisons.get(c, {}).get("max_overlap", 0) for c in other_classes]
    mr_min_overlaps = [mr_comparisons.get(c, {}).get("min_overlap", 0) for c in other_classes]
    
    # Create the plot with two rows - one for all regions, one for MR
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    x = np.arange(len(other_classes))
    width = 0.2  # Width of bars
    
    # Plot data for all regions in the first subplot
    avg_bars1 = ax1.bar(x - 1.5*width, avg_overlaps, width, color='blue', label='Average Overlap')
    max_bars1 = ax1.bar(x - 0.5*width, max_overlaps, width, color='green', label='Max Overlap')
    min_bars1 = ax1.bar(x + 0.5*width, min_overlaps, width, color='red', label='Min Overlap')
    int_bars1 = ax1.bar(x + 1.5*width, intersection_overlaps, width, color='purple', label='All Regions Intersection')
    
    # Plot data for MR only in the second subplot
    avg_bars2 = ax2.bar(x - width, mr_avg_overlaps, width, color='blue', label='MR Avg Overlap')
    max_bars2 = ax2.bar(x, mr_max_overlaps, width, color='green', label='MR Max Overlap')
    min_bars2 = ax2.bar(x + width, mr_min_overlaps, width, color='red', label='MR Min Overlap')
    
    # Add numeric labels on top of each bar
    def add_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                ax.text(bar.get_x() + bar.get_width()/2, height + 3,
                      f'{int(height) if float(height).is_integer() else round(float(height), 1)}',
                      ha='center', va='bottom', fontsize=8)
    
    # Add labels to all bars
    add_labels(ax1, avg_bars1)
    add_labels(ax1, max_bars1)
    add_labels(ax1, min_bars1)
    add_labels(ax1, int_bars1)
    
    add_labels(ax2, avg_bars2)
    add_labels(ax2, max_bars2)
    add_labels(ax2, min_bars2)
    
    # Set titles and labels
    ax1.set_title(f"All Regions of Class {target_class} vs Other Classes")
    ax2.set_title(f"Only Major Region of Class {target_class} vs Other Classes")
    
    ax2.set_xlabel("Other Classes")
    ax1.set_ylabel("Coordinates Overlap")
    ax2.set_ylabel("Coordinates Overlap")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(other_classes, rotation=45)
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Add summary text
    overall_summary = comparison_results.get("summary", {})
    if overall_summary:
        summary_text = (
            f"Overall: Avg={overall_summary.get('avg_overlap', 'N/A')}, "
            f"Max={overall_summary.get('max_overlap', 'N/A')}, "
            f"Min={overall_summary.get('min_overlap', 'N/A')}, "
            f"Total Comparisons={overall_summary.get('total_comparisons', 'N/A')}"
        )
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=9)
    
    # Adjust plot layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.3)


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

    # Generate MR vs Extra Regions plot
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
    
    print(f"Main analysis plot saved to: {plot_path}")

    # Generate per-class vs others plots
    print("Generating per-class vs others comparison plots...")
    class_vs_others_dir = os.path.join(folder_path, "class_vs_others_plots")
    for class_label in major_regions.keys():
        comparison_vs_others = compare_class_against_others(class_label, major_regions, unique_patterns)
        plot_class_vs_others(class_label, comparison_vs_others, class_vs_others_dir)

    print(f"Per-class comparison plots saved in: {class_vs_others_dir}")


if __name__ == "__main__":
    main()