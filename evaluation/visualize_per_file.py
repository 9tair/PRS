import json
import matplotlib.pyplot as plt
import os
import argparse

def replace_none_with_previous(lst):
    """Replaces None values with the previous non-None value in the list."""
    if not lst:
        return []
    
    cleaned_list = []
    prev_value = 0  # Default to 0 if the first value is None

    for val in lst:
        if val is None:
            cleaned_list.append(prev_value)  # Use previous value
        else:
            cleaned_list.append(val)
            prev_value = val  # Update previous value

    return cleaned_list

def extract_info_from_path(filepath):
    """Extracts dataset name, batch size, and warmup info from the given path."""
    try:
        parts = filepath.split(os.sep)
        parent_folder = parts[-2]  # Folder containing the metrics file

        # Example: "VGG16_CIFAR10_batch_2048_warmup_50"
        if "batch_" in parent_folder:
            model_info = parent_folder.split("_batch_")
            model_dataset = model_info[0]  # Example: "VGG16_CIFAR10"
            batch_warmup = model_info[1]  # Example: "2048_warmup_50"
            
            batch_size = batch_warmup.split("_warmup_")[0]  # Extract batch size
            return model_dataset, batch_size
    except Exception as e:
        print(f"Error extracting dataset and batch size from path: {e}")
    
    return "Unknown", "Unknown"

def visualize_single_metric(filepath):
    """Visualizes metrics from a specific JSON file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # Load results
    with open(filepath, "r") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file {filepath}. Please check the file format.")
            return

    # Check if necessary keys exist and contain valid data
    required_keys = ["epoch", "train_accuracy", "test_accuracy", "prs_ratios"]
    for key in required_keys:
        if key not in metrics:
            print(f"Error: Missing key '{key}' in {filepath}.")
            return
        if metrics[key] is None:
            print(f"Error: '{key}' is None in {filepath}.")
            return
        if not isinstance(metrics[key], list):
            print(f"Error: '{key}' is not a list in {filepath}.")
            return

    # Extract values and replace None with the previous number
    epochs = metrics["epoch"]
    train_accuracy = replace_none_with_previous(metrics["train_accuracy"])
    test_accuracy = replace_none_with_previous(metrics["test_accuracy"])
    prs_ratios = replace_none_with_previous(metrics["prs_ratios"])

    # Normalize train and test accuracy
    train_accuracy = [acc / 100 for acc in train_accuracy]
    test_accuracy = [acc / 100 for acc in test_accuracy]

    # Extract dataset name and batch size from filepath
    dataset_name, batch_size = extract_info_from_path(filepath)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracy, label="Train Accuracy", linewidth=1.5, color="blue")
    plt.plot(epochs, test_accuracy, label="Test Accuracy", linewidth=1.5, color="orange")
    plt.plot(epochs, prs_ratios, label="PRS Ratio", linewidth=1.5, color="green")

    plt.xlabel("Epochs")
    plt.ylabel("Normalized Value")
    plt.title(f"{dataset_name} - Batch {batch_size}")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(range(0, len(epochs) + 1, 50))  # Adjust x-axis visibility
    plt.ylim(0, 1)  # Keep y-axis consistent

    # Save plot in the same directory as the metrics file
    save_path = os.path.join(os.path.dirname(filepath), "visualization_metrics.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Use argparse to accept filepath as a command-line argument
    parser = argparse.ArgumentParser(description="Visualize training metrics from a JSON file.")
    parser.add_argument("--file", type=str, required=True, help="Full file path to the metrics JSON file.")

    args = parser.parse_args()
    
    # Call function with the provided file path
    visualize_single_metric(args.file)
