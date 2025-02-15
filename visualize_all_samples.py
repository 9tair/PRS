import json
import matplotlib.pyplot as plt
import os
from config import config

def visualize_metrics(filename):
    """Visualizes metrics from a specific JSON file."""
    metrics_path = os.path.join(config['results_save_path'], filename)
    
    if not os.path.exists(metrics_path):
        print(f"File not found: {metrics_path}")
        return

    # Load results
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    epochs = metrics.get("epoch", [])
    train_accuracy = [acc / 100 for acc in metrics.get("train_accuracy", [])]
    test_accuracy = [acc / 100 for acc in metrics.get("test_accuracy", [])]
    prs_ratios = metrics.get("prs_ratios", [])

    if not epochs:
        print(f"No epoch data found in {filename}")
        return

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
    plt.xticks(range(0, len(epochs) + 1, max(1, len(epochs) // 10)))  # Adjust x-axis visibility
    plt.ylim(0, 1)  # Keep y-axis consistent

    # Save plot
    save_path = os.path.join(config['results_save_path'], f"visualization_{dataset_name}_batch_{batch_size}.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Saved: {save_path}")

if __name__ == "__main__":
    results_path = config['results_save_path']
    json_files = [f for f in os.listdir(results_path) if f.startswith("metrics_") and f.endswith(".json")]
    
    if not json_files:
        print("No metric files found in the results folder.")
    else:
        for file in json_files:
            visualize_metrics(file)
