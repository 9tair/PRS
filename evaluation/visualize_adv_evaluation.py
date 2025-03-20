import os
import sys
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

# Import project-specific modules
from config import config

def load_results(file_path):
    """Load results from HDF5 file."""
    results = {}
    
    with h5py.File(file_path, "r") as hf:
        for model_name in hf.keys():
            results[model_name] = {}
            model_group = hf[model_name]
            
            for attack_name in model_group.keys():
                results[model_name][attack_name] = {}
                attack_group = model_group[attack_name]
                
                for epsilon in attack_group.keys():
                    results[model_name][attack_name][epsilon] = {}
                    eps_group = attack_group[epsilon]
                    
                    # Directly handle Train and Test data
                    for dataset_type in ['Train', 'Test']:
                        # Check if this dataset type exists in the HDF5 structure
                        if dataset_type in eps_group:
                            dataset_group = eps_group[dataset_type]
                            results[model_name][attack_name][epsilon][dataset_type] = {}
                            
                            # Load regular datasets
                            for key in dataset_group.keys():
                                results[model_name][attack_name][epsilon][dataset_type][key] = dataset_group[key][()]
                            
                            # Load attributes (for JSON strings)
                            for attr_key in dataset_group.attrs.keys():
                                try:
                                    attr_value = json.loads(dataset_group.attrs[attr_key])
                                    results[model_name][attack_name][epsilon][dataset_type][attr_key] = attr_value
                                except:
                                    results[model_name][attack_name][epsilon][dataset_type][attr_key] = dataset_group.attrs[attr_key]
                        else:
                            # Alternative structure: the keys might be data items directly
                            try:
                                # Try to parse this as a JSON attribute
                                if dataset_type in eps_group.attrs:
                                    results[model_name][attack_name][epsilon][dataset_type] = json.loads(eps_group.attrs[dataset_type])
                            except (KeyError, json.JSONDecodeError):
                                # If no Train/Test structure exists, we might have a different format
                                # Just take whatever is available and log a warning
                                print(f"Warning: Dataset type {dataset_type} not found in {model_name}/{attack_name}/{epsilon}")
                                results[model_name][attack_name][epsilon][dataset_type] = {}
    
    return results

def plot_accuracy_comparison(results, output_dir):
    """Plot accuracy comparison across models, attacks and epsilons."""
    # Get unique values for models, attacks and epsilons
    model_names = list(results.keys())
    
    # Ensure we have at least one model
    if not model_names:
        print("Error: No models found in results")
        return
    
    attacks = list(results[model_names[0]].keys())
    
    # Ensure we have at least one attack
    if not attacks:
        print("Error: No attacks found in results")
        return
    
    # Get all epsilons (could be different for each attack/model)
    all_epsilons = set()
    for model in model_names:
        for attack in attacks:
            all_epsilons.update(results[model][attack].keys())
    
    # Filter out non-numeric epsilons and sort
    epsilons = sorted([eps for eps in all_epsilons if eps.replace('.', '', 1).isdigit()], 
                      key=lambda x: float(x))
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot for Train dataset
    ax = axes[0]
    for model_idx, model_name in enumerate(model_names):
        for attack_idx, attack in enumerate(attacks):
            # Collect values for this model/attack combination
            epsilon_values = []
            accuracy_values = []
            
            for eps in epsilons:
                # Skip if this epsilon doesn't exist for this model/attack
                if eps not in results[model_name][attack]:
                    continue
                
                # Skip if Train data doesn't exist for this epsilon
                if 'Train' not in results[model_name][attack][eps]:
                    continue
                
                # Add data point
                epsilon_values.append(float(eps))
                accuracy_values.append(results[model_name][attack][eps]['Train'].get('accuracy', 0))
            
            # Skip if no data points
            if not epsilon_values:
                continue
            
            # Plot
            line_style = '-' if model_idx == 0 else '--'
            marker = ['o', 's', '^'][attack_idx % 3]  # Use modulo for safety
            color = plt.cm.tab10(model_idx * len(attacks) + attack_idx)
            
            ax.plot(epsilon_values, accuracy_values, 
                    marker=marker, linestyle=line_style, color=color,
                    label=f"{model_name} - {attack}")
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Attack Robustness - Training Set')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Plot for Test dataset
    ax = axes[1]
    for model_idx, model_name in enumerate(model_names):
        for attack_idx, attack in enumerate(attacks):
            # Collect values for this model/attack combination
            epsilon_values = []
            accuracy_values = []
            
            for eps in epsilons:
                # Skip if this epsilon doesn't exist for this model/attack
                if eps not in results[model_name][attack]:
                    continue
                
                # Skip if Test data doesn't exist for this epsilon
                if 'Test' not in results[model_name][attack][eps]:
                    continue
                
                # Add data point
                epsilon_values.append(float(eps))
                accuracy_values.append(results[model_name][attack][eps]['Test'].get('accuracy', 0))
            
            # Skip if no data points
            if not epsilon_values:
                continue
            
            # Plot
            line_style = '-' if model_idx == 0 else '--'
            marker = ['o', 's', '^'][attack_idx % 3]  # Use modulo for safety
            color = plt.cm.tab10(model_idx * len(attacks) + attack_idx)
            
            ax.plot(epsilon_values, accuracy_values, 
                    marker=marker, linestyle=line_style, color=color,
                    label=f"{model_name} - {attack}")
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Attack Robustness - Test Set')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add legend to the right of the plots
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_misclassification_matrix(results, output_dir):
    """Plot misclassification matrices for each model and attack."""
    model_names = list(results.keys())
    
    for model_name in model_names:
        attacks = list(results[model_name].keys())
        
        for attack in attacks:
            epsilons = list(results[model_name][attack].keys())
            
            for eps in epsilons:
                # Skip if Test data doesn't exist for this epsilon
                if 'Test' not in results[model_name][attack][eps]:
                    continue
                
                # Get misclassification data
                misclassification_data = results[model_name][attack][eps]['Test'].get('misclassification_counts', {})
                
                if not misclassification_data:
                    continue
                
                # Extract all classes present
                all_classes = set()
                for true_class in misclassification_data:
                    all_classes.add(int(true_class))
                    for pred_class in misclassification_data[true_class]:
                        all_classes.add(int(pred_class))
                
                # Make sure we have at least one class
                if not all_classes:
                    continue
                
                num_classes = max(all_classes) + 1
                
                # Create confusion matrix
                confusion_matrix = np.zeros((num_classes, num_classes))
                
                for true_class, pred_dict in misclassification_data.items():
                    for pred_class, count in pred_dict.items():
                        confusion_matrix[int(true_class), int(pred_class)] = count
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
                plt.xlabel('Predicted Class')
                plt.ylabel('True Class')
                plt.title(f'Misclassification Matrix - {model_name} - {attack} (ε={eps})')
                
                # Sanitize filename
                safe_model_name = model_name.replace('/', '_').replace('\\', '_')
                save_path = os.path.join(output_dir, f'misclassification_{safe_model_name}_{attack}_eps{eps}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

def plot_logit_shifts(results, output_dir):
    """Plot distribution of logit shifts for each model and attack."""
    model_names = list(results.keys())
    num_models = len(model_names)
    
    if num_models == 0:
        print("Error: No models found in results")
        return
    
    # Get a list of all attacks across all models
    all_attacks = set()
    for model_name in model_names:
        all_attacks.update(results[model_name].keys())
    attacks = sorted(list(all_attacks))
    num_attacks = len(attacks)
    
    if num_attacks == 0:
        print("Error: No attacks found in results")
        return
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_models, num_attacks, figsize=(15, 5 * num_models), squeeze=False)
    
    for i, model_name in enumerate(model_names):
        for j, attack in enumerate(attacks):
            ax = axes[i, j]
            
            # Skip if this attack doesn't exist for this model
            if attack not in results[model_name]:
                ax.set_visible(False)
                continue
            
            epsilons = list(results[model_name][attack].keys())
            
            for eps_idx, eps in enumerate(epsilons):
                # Skip if Test data doesn't exist for this epsilon
                if 'Test' not in results[model_name][attack][eps]:
                    continue
                
                logit_shifts = results[model_name][attack][eps]['Test'].get('logit_shifts', [])
                
                if not logit_shifts:
                    continue
                
                # Create density plot
                sns.kdeplot(logit_shifts, ax=ax, label=f'ε={eps}')
            
            ax.set_xlabel('Logit Shift Magnitude')
            ax.set_ylabel('Density')
            
            # Set titles for first row and column only
            if i == 0:
                ax.set_title(attack)
            if j == 0:
                ax.text(-0.2, 0.5, model_name, va='center', ha='center', 
                         rotation=90, transform=ax.transAxes, fontsize=12)
            
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logit_shifts.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_gradient_norms(results, output_dir):
    """Plot distribution of gradient norms for each model and attack."""
    model_names = list(results.keys())
    num_models = len(model_names)
    
    if num_models == 0:
        print("Error: No models found in results")
        return
    
    # Get a list of all attacks across all models
    all_attacks = set()
    for model_name in model_names:
        all_attacks.update(results[model_name].keys())
    attacks = sorted(list(all_attacks))
    num_attacks = len(attacks)
    
    if num_attacks == 0:
        print("Error: No attacks found in results")
        return
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_models, num_attacks, figsize=(15, 5 * num_models), squeeze=False)
    
    for i, model_name in enumerate(model_names):
        for j, attack in enumerate(attacks):
            ax = axes[i, j]
            
            # Skip if this attack doesn't exist for this model
            if attack not in results[model_name]:
                ax.set_visible(False)
                continue
            
            epsilons = list(results[model_name][attack].keys())
            
            for eps_idx, eps in enumerate(epsilons):
                # Skip if Test data doesn't exist for this epsilon
                if 'Test' not in results[model_name][attack][eps]:
                    continue
                
                gradient_norms = results[model_name][attack][eps]['Test'].get('gradient_norms', [])
                
                if not gradient_norms:
                    continue
                
                # Create density plot
                sns.kdeplot(gradient_norms, ax=ax, label=f'ε={eps}')
            
            ax.set_xlabel('Gradient Norm')
            ax.set_ylabel('Density')
            
            # Set titles for first row and column only
            if i == 0:
                ax.set_title(attack)
            if j == 0:
                ax.text(-0.2, 0.5, model_name, va='center', ha='center', 
                         rotation=90, transform=ax.transAxes, fontsize=12)
            
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_norms.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_region_analysis(results, output_dir):
    """Plot analysis of fooling rates by region type (major vs extra)."""
    model_names = list(results.keys())
    num_models = len(model_names)
    
    if num_models == 0:
        print("Error: No models found in results")
        return
    
    # Create a figure with subplots for each model
    fig, axes = plt.subplots(num_models, 1, figsize=(12, 8 * num_models), squeeze=False)
    
    for i, model_name in enumerate(model_names):
        ax = axes[i, 0]
        
        attacks = list(results[model_name].keys())
        
        # Data for the stacked bar chart
        x_positions = []
        x_labels = []
        major_region_data = []
        extra_region_data = []
        
        position_counter = 0
        for attack_idx, attack in enumerate(attacks):
            epsilons = list(results[model_name][attack].keys())
            
            for eps_idx, eps in enumerate(epsilons):
                # Skip if Test data doesn't exist for this epsilon
                if 'Test' not in results[model_name][attack][eps]:
                    continue
                
                x_positions.append(position_counter)
                position_counter += 1
                x_labels.append(f"{attack}\nε={eps}")
                
                # Get region data
                major_region_fooled = results[model_name][attack][eps]['Test'].get('major_region_fooled', 0)
                extra_region_fooled = results[model_name][attack][eps]['Test'].get('extra_region_fooled', 0)
                
                major_region_data.append(major_region_fooled)
                extra_region_data.append(extra_region_fooled)
        
        # Skip if no data
        if not x_positions:
            ax.set_visible(False)
            continue
        
        # Create stacked bar chart
        width = 0.8
        ax.bar(x_positions, major_region_data, width, label='Major Region')
        ax.bar(x_positions, extra_region_data, width, bottom=major_region_data, label='Extra Region')
        
        # Set labels and title
        ax.set_ylabel('Number of Samples Fooled')
        ax.set_title(f'Region Analysis for {model_name}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'region_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(results, output_dir):
    """Create a summary table of key results."""
    # Initialize list to hold summary data
    summary_data = []
    
    # Iterate through all models, attacks, and epsilons
    for model_name, model_data in results.items():
        for attack_name, attack_data in model_data.items():
            for eps, eps_data in attack_data.items():
                # Skip if Test data doesn't exist for this epsilon
                if 'Test' not in eps_data:
                    continue
                
                test_data = eps_data['Test']
                
                # Extract key metrics
                accuracy = test_data.get('accuracy', 0)
                total_fooled = test_data.get('total_fooled', 0)
                major_region_fooled = test_data.get('major_region_fooled', 0)
                extra_region_fooled = test_data.get('extra_region_fooled', 0)
                
                # Calculate average logit shift and gradient norm
                logit_shifts = test_data.get('logit_shifts', [])
                gradient_norms = test_data.get('gradient_norms', [])
                
                avg_logit_shift = np.mean(logit_shifts) if logit_shifts else 0
                avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0
                
                # Add row to summary data
                summary_data.append({
                    'Model': model_name,
                    'Attack': attack_name,
                    'Epsilon': eps,
                    'Accuracy (%)': round(accuracy, 2),
                    'Total Fooled': total_fooled,
                    'Major Region Fooled': major_region_fooled,
                    'Extra Region Fooled': extra_region_fooled,
                    'Avg Logit Shift': round(avg_logit_shift, 4),
                    'Avg Gradient Norm': round(avg_gradient_norm, 4)
                })
    
    # Check if we have any data
    if not summary_data:
        print("Error: No valid data found for summary report")
        return
    
    # Write the summary to a CSV file
    import csv
    
    with open(os.path.join(output_dir, 'summary_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"Summary report saved to {os.path.join(output_dir, 'summary_results.csv')}")

def main():
    """Main function for visualizing adversarial robustness results."""
    import argparse
    parser = argparse.ArgumentParser(description="Visualize adversarial robustness results.")
    parser.add_argument("--results_file", type=str, required=False, 
                        default=os.path.join(config["results_save_path"], "adversarial_results.h5"),
                        help="Path to the HDF5 results file.")
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found at {args.results_file}")
        return
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    
    # Create output directory (same folder as results file)
    output_dir = os.path.dirname(args.results_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating accuracy comparison plots...")
    plot_accuracy_comparison(results, output_dir)
    
    print("Generating misclassification matrices...")
    plot_misclassification_matrix(results, output_dir)
    
    print("Generating logit shift distributions...")
    plot_logit_shifts(results, output_dir)
    
    print("Generating gradient norm distributions...")
    plot_gradient_norms(results, output_dir)
    
    print("Generating region analysis plots...")
    plot_region_analysis(results, output_dir)
    
    print("Creating summary report...")
    create_summary_report(results, output_dir)
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()