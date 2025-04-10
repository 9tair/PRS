import os
import json
import argparse
import numpy as np
from collections import Counter, defaultdict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_sample_to_class_map(major_regions):
    sample_to_class = {}
    for class_key, class_data in major_regions.items():
        class_id = int(class_key.split("_")[-1])

        # Add major region samples
        if "samples" in class_data["major_region"]:
            for idx in class_data["major_region"]["samples"]:
                sample_to_class[idx] = class_id

        # Add extra region samples
        for er in class_data.get("extra_regions", []):
            if "samples" in er:
                for idx in er["samples"]:
                    sample_to_class[idx] = class_id
    return sample_to_class

def analyze_region(region_id, region_type, unique_patterns, sample_to_class):
    pattern_data = unique_patterns.get(str(region_id))
    if not pattern_data:
        return None

    samples = pattern_data.get("samples", [])
    raw_activations = np.array(pattern_data.get("original_activations", []))

    if raw_activations.size == 0:
        return None

    # Get class labels for the samples
    region_classes = [sample_to_class.get(sid, -1) for sid in samples]
    region_classes = [c for c in region_classes if c >= 0]
    class_distribution = dict(Counter(region_classes))

    # Analyze non-zero neuron stats
    non_zero = raw_activations[raw_activations != 0]
    if non_zero.size == 0:
        stats = {
            "mean": None, "min": None, "max": None, "std": None
        }
        avg_active = 0.0
    else:
        stats = {
            "mean": float(np.mean(non_zero)),
            "min": float(np.min(non_zero)),
            "max": float(np.max(non_zero)),
            "std": float(np.std(non_zero))
        }
        avg_active = float(np.mean(np.count_nonzero(raw_activations, axis=1)))

    return {
        "activation_index": int(region_id),
        "region_type": region_type,
        "num_samples": len(samples),
        "class_distribution": class_distribution,
        "nonzero_neuron_stats": stats,
        "avg_active_neurons_per_sample": avg_active
    }

def analyze_all_regions(major_regions, unique_patterns, sample_to_class):
    results = {}

    for class_key, class_data in major_regions.items():
        class_id = int(class_key.split("_")[-1])
        results[class_key] = []

        # Analyze major region
        major_index = class_data["major_region"]["activation_index"]
        major_result = analyze_region(major_index, "major", unique_patterns, sample_to_class)
        if major_result:
            results[class_key].append(major_result)

        # Analyze extra (marginal) regions
        for er in class_data.get("extra_regions", []):
            extra_index = er["activation_index"]
            extra_result = analyze_region(extra_index, "extra", unique_patterns, sample_to_class)
            if extra_result:
                results[class_key].append(extra_result)

    return results

def main(checkpoint_dir):
    output_dir = os.path.join(checkpoint_dir, "polysemanticity_analysis")
    os.makedirs(output_dir, exist_ok=True)

    major_regions_path = os.path.join(checkpoint_dir, "major_regions.json")
    unique_patterns_path = os.path.join(checkpoint_dir, "unique_patterns.json")
    assert os.path.exists(major_regions_path), "Missing major_regions.json"
    assert os.path.exists(unique_patterns_path), "Missing unique_patterns.json"

    print("📦 Loading data...")
    major_regions = load_json(major_regions_path)
    unique_patterns = load_json(unique_patterns_path)
    sample_to_class = build_sample_to_class_map(major_regions)

    print("🔍 Analyzing all regions (major + extra)...")
    stats = analyze_all_regions(major_regions, unique_patterns, sample_to_class)

    output_path = os.path.join(output_dir, "full_region_analysis.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f" Saved full region analysis to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Major + Extra Regions")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to model's epoch_X folder")
    args = parser.parse_args()
    main(args.checkpoint_dir)
