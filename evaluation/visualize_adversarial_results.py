import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Constants
RESULTS_PATH = "results/adversarial_results.json"
SAVE_PATH = "results/adversarial_results_plot.png"

# Define colors and styles dynamically
COLOR_PALETTE = {"Simple": "darkorange", "PRS": "royalblue"}
LINE_STYLES = {"Train": "solid", "Test": "dashed"}
MARKERS = {"Train": "o", "Test": "s"}

def load_results():
    """Loads the adversarial robustness results from JSON file."""
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"❌ Results file not found: {RESULTS_PATH}")

    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    print("✅ Successfully Loaded Results!")
    return results

def plot_adversarial_results(results):
    """
    Plots robust accuracy vs. epsilon for different adversarial attacks.
    
    Args:
        results (dict): Dictionary of adversarial accuracy results.
    """
    model_names = list(results.keys())  # Extract models dynamically
    attack_names = list(next(iter(results.values())).keys())  # Extract attack types
    epsilons = sorted([float(e) for e in next(iter(results.values())).get(next(iter(attack_names)), {}).get("Train", {}).keys()])  # Extract valid epsilon values

    num_attacks = len(attack_names)
    fig, axes = plt.subplots(1, num_attacks, figsize=(4 * num_attacks, 5), sharey=True)

    if num_attacks == 1:
        axes = [axes]

    handles = []  # Store handles for legend
    labels = []   # Store labels for legend

    for i, attack in enumerate(attack_names):
        ax = axes[i]
        has_data = False  # Flag to check if attack has any valid data

        for model in model_names:
            model_type = "Simple" if "Simple" in model else "PRS"

            for eval_type in ["Train", "Test"]:
                color = COLOR_PALETTE[model_type]
                linestyle = LINE_STYLES[eval_type]
                marker = MARKERS[eval_type]
                label = f"{model} ({eval_type})"

                # Ensure attack exists in the JSON
                if attack not in results[model]:
                    print(f"⚠️ Missing attack: {attack} for model: {model}. Skipping...")
                    continue

                try:
                    acc_values = [results[model][attack][eval_type].get(str(eps), 0) / 100 for eps in epsilons]
                    
                    if all(v == 0 for v in acc_values):  # If all values are 0, skip
                        print(f"⚠️ No valid data for {model} - {attack} ({eval_type})")
                        continue

                    has_data = True  # Mark that this attack has data

                    line, = ax.plot(
                        epsilons,
                        acc_values,
                        marker=marker,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        linewidth=2,
                        markersize=6
                    )

                    # Add legend handle only once per model-eval pair
                    if label not in labels:
                        handles.append(line)
                        labels.append(label)

                except KeyError as e:
                    print(f"⚠️ KeyError: {e} - Check if JSON structure is correct for {model} under {attack} ({eval_type})!")

        # If no data was plotted for this attack, show warning
        if not has_data:
            print(f"⚠️ No data available for {attack}. Skipping subplot...")

        ax.set_title(f"{attack} Attack", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epsilon (ε)", fontsize=12)
        ax.set_xlim(min(epsilons), max(epsilons))
        ax.set_ylim(0, 1.0)
        ax.set_xticks(epsilons)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Robust Accuracy", fontsize=12)

    # 🔹 Add a proper legend BELOW the plot
    fig.legend(handles, labels, loc="lower center", fontsize=12, ncol=2, bbox_to_anchor=(0.5, -0.12))

    # Save and show plot
    os.makedirs("results", exist_ok=True)
    plt.tight_layout(rect=[0, 0.15, 1, 1])  # Adjusted for legend space
    plt.savefig(SAVE_PATH, dpi=300)
    plt.show()

    print(f"\n📊 Plot saved to: {SAVE_PATH}")

def main():
    """Main function to load results and generate visualization."""
    results = load_results()
    plot_adversarial_results(results)

if __name__ == "__main__":
    main()
