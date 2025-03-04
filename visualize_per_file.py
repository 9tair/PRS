import json
import matplotlib.pyplot as plt
import os
from config import config

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

def visualize_single_metric(filename):
    """Visualizes metrics from a specific JSON file."""
    metrics_path = os.path.join(config['results_save_path'], filename)
    
    if not os.path.exists(metrics_path):
        print(f"File not found: {metrics_path}")
        return

    # Load results
    with open(metrics_path, "r") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file {filename}. Please check the file format.")
            return

    # Check if necessary keys exist and contain valid data
    required_keys = ["epoch", "train_accuracy", "test_accuracy", "prs_ratios"]
    for key in required_keys:
        if key not in metrics:
            print(f"Error: Missing key '{key}' in {filename}.")
            return
        if metrics[key] is None:
            print(f"Error: '{key}' is None in {filename}.")
            return
        if not isinstance(metrics[key], list):
            print(f"Error: '{key}' is not a list in {filename}.")
            return

    # Extract values and replace None with the previous number
    epochs = metrics["epoch"]
    train_accuracy = replace_none_with_previous(metrics["train_accuracy"])
    test_accuracy = replace_none_with_previous(metrics["test_accuracy"])
    prs_ratios = replace_none_with_previous(metrics["prs_ratios"])

    # Normalize train and test accuracy
    train_accuracy = [acc / 100 for acc in train_accuracy]
    test_accuracy = [acc / 100 for acc in test_accuracy]

    # Extract dataset name and batch size from filename
    try:
        parts = filename.replace("metrics_", "").replace(".json", "").split("_batch_")
        dataset_name, batch_size = parts[0], parts[1]
    except IndexError:
        dataset_name, batch_size = "Unknown", "Unknown"

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

    # Save plot
    save_path = os.path.join(config['results_save_path'], f"visualization_{dataset_name}_batch_{batch_size}.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Saved: {save_path}")

if __name__ == "__main__":
    filename = input("Enter the filename (e.g., metrics_CIFAR10_batch_32.json): ").strip()
    visualize_single_metric(filename)
