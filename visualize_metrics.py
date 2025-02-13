import json
import matplotlib.pyplot as plt
import os
from config import config

def visualize_metrics():
    """Visualizes all metrics stored in separate files."""
    datasets = ["CIFAR10", "MNIST", "F-MNIST"]
    
    for dataset_name in datasets:
        for batch_size in config["batch_sizes"]:
            metrics_path = f"{config['results_save_path']}metrics_{dataset_name}_batch_{batch_size}.json"
            
            if not os.path.exists(metrics_path):
                print(f"Skipping {metrics_path} (Not Found)")
                continue

            # Load results
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            epochs = metrics["epoch"]
            train_accuracy = [acc / 100 for acc in metrics["train_accuracy"]]
            test_accuracy = [acc / 100 for acc in metrics["test_accuracy"]]
            prs_ratios = metrics["prs_ratios"]

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
            plt.xticks(range(0, len(epochs)+1, 50))  # Adjust x-axis visibility
            plt.ylim(0, 1)  # Keep y-axis consistent

            # Save plot
            save_path = f"{config['results_save_path']}visualization_{dataset_name}_batch_{batch_size}.png"
            plt.savefig(save_path)
            plt.show()

            print(f"Saved: {save_path}")

if __name__ == "__main__":
    visualize_metrics()
