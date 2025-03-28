import os
import json
import argparse
import matplotlib.pyplot as plt

from datetime import datetime


# Utility functions
def parse_summary(path):
    summary_path = os.path.join(path, "attacks", "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found at: {summary_path}")

    with open(summary_path, "r") as f:
        return json.load(f)


def extract_model_label(path):
    """Create a unique label for the model based on folder path"""
    parts = os.path.normpath(path).split(os.sep)
    try:
        model_part = parts[-2]  # like 'CNN-6_CIFAR10_batch_128'
        epoch_part = parts[-1]  # like 'epoch_300'
        return f"{model_part}_{epoch_part}"
    except IndexError:
        return os.path.basename(path)


def format_save_name(label1, label2):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/comparison_{label1}_vs_{label2}_{timestamp}.png"


def plot_comparison(summary1, summary2, label1, label2, save_path):
    attacks = summary1["accuracy_results"].keys() | summary2["accuracy_results"].keys()
    attacks = sorted(attacks)

    num_attacks = len(attacks)
    fig, axes = plt.subplots(1, num_attacks, figsize=(5 * num_attacks, 5), sharey=True)

    if num_attacks == 1:
        axes = [axes]

    handles = []
    labels = []

    for i, attack in enumerate(attacks):
        ax = axes[i]

        for summary, label, color in zip([summary1, summary2], [label1, label2], ["darkorange", "royalblue"]):
            results = summary["accuracy_results"].get(attack)

            if results is None:
                continue

            epsilons = []
            train_accuracies = []
            test_accuracies = []

            # Handle attacks with a single summary value (like CW)
            if isinstance(results, dict) and "test_accuracy" in results:
                epsilons = ["default"]
                train_accuracies = [results["train_accuracy"] / 100]
                test_accuracies = [results["test_accuracy"] / 100]
            else:
                epsilons = sorted([float(e) for e in results.keys()])
                train_accuracies = [results[str(e)]["train_accuracy"] / 100 for e in epsilons]
                test_accuracies = [results[str(e)]["test_accuracy"] / 100 for e in epsilons]

            # Plot train
            h1, = ax.plot(
                epsilons,
                train_accuracies,
                label=f"{label} (Train)",
                color=color,
                linestyle="solid",
                marker="o"
            )

            # Plot test
            h2, = ax.plot(
                epsilons,
                test_accuracies,
                label=f"{label} (Test)",
                color=color,
                linestyle="dashed",
                marker="s"
            )

            if f"{label} (Train)" not in labels:
                handles.extend([h1, h2])
                labels.extend([f"{label} (Train)", f"{label} (Test)"])

        ax.set_title(f"{attack} Attack", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epsilon (ε)" if epsilons[0] != "default" else "Default Attack", fontsize=12)
        if epsilons[0] != "default":
            ax.set_xticks(epsilons)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Robust Accuracy", fontsize=12)
    fig.legend(handles, labels, loc="lower center", fontsize=12, ncol=2, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"\nVisualization saved at: {save_path}")


# Main script
def main():
    parser = argparse.ArgumentParser(description="Compare adversarial results of two models")
    parser.add_argument("--folder1", type=str, required=True, help="Path to first model folder")
    parser.add_argument("--folder2", type=str, required=True, help="Path to second model folder")
    args = parser.parse_args()

    summary1 = parse_summary(args.folder1)
    summary2 = parse_summary(args.folder2)

    label1 = extract_model_label(args.folder1)
    label2 = extract_model_label(args.folder2)

    save_path = format_save_name(label1, label2)
    plot_comparison(summary1, summary2, label1, label2, save_path)


if __name__ == "__main__":
    main()
