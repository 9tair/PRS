import sys
import os
import re

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Import with correct paths for your project structure
from config import config
from utils import fgsm_attack, bim_attack, pgd_attack, get_datasets
from models.model_factory import get_model  # Assuming this is the correct path based on error

# Define attack configurations with multiple perturbations
EPSILONS = [0.0, 0.0313, 0.05, 0.1]  # Different epsilon values for robustness analysis

ATTACKS = {
    "FGSM": {"attack_fn": fgsm_attack, "params": {}},  # Epsilon will be varied dynamically
    "BIM": {"attack_fn": bim_attack, "params": {"alpha": 0.01, "num_iter": 10}},  # BIM uses alpha
    "PGD-20": {"attack_fn": pgd_attack, "params": {"alpha": 0.007, "num_iter": 20}},  # PGD 20 steps
    "PGD-100": {"attack_fn": pgd_attack, "params": {"alpha": 0.007, "num_iter": 100}},  # PGD 100 steps
}

def evaluate_robustness(model, data_loader, attack_fn, attack_params, epsilon, device="cuda"):
    """
    Evaluates the model robustness under a given adversarial attack at a specific epsilon.
    """
    model.to(device).eval()
    correct, total = 0, 0

    for inputs, labels in tqdm(data_loader, desc=f"Running {attack_fn.__name__} (ε={epsilon})"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Use clean inputs when ε=0
        if epsilon == 0:
            adv_inputs = inputs
        else:
            adv_inputs = attack_fn(model, inputs, labels, epsilon=epsilon, **attack_params)

        # Get model predictions
        outputs = model(adv_inputs)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return correct / total * 100  # Return accuracy as percentage

def extract_model_info(folder_path):
    """
    Extract model parameters from folder path, handling both formats:
    - /home/tair/project_root/models/saved/CNN-6_CIFAR10_batch_128_warmup_50_PRS
    - /home/tair/project_root/models/saved/CNN-6_CIFAR10_batch_128/epoch_300
    - /home/tair/project_root/models/saved/CNN-6_CIFAR10_batch_128_warmup_50_PRS/epoch_63
    
    Returns model_name, dataset_name, batch_size, prs_enabled, warmup_epochs, epoch_num
    """
    path = Path(folder_path)
    
    # Check if we're in an epoch subdirectory
    epoch_num = None
    if path.name.startswith('epoch_'):
        # Extract epoch number
        epoch_match = re.search(r'epoch_(\d+)', path.name)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
        # Use parent directory for model info
        model_dir = path.parent.name
    else:
        # We're directly in the model directory
        model_dir = path.name
    
    # Split the model directory name
    parts = model_dir.split('_')
    
    # Extract basic information
    model_name = parts[0]  # e.g., 'CNN-6'
    
    # For the dataset name, we need to be careful as it might contain hyphens
    # Assuming dataset is always followed by 'batch'
    dataset_name = None
    for i, part in enumerate(parts):
        if part == "batch" and i > 0:
            # The part before "batch" is the dataset name
            dataset_name = parts[i-1]
            break
    
    if dataset_name is None:
        raise ValueError(f"Could not extract dataset name from folder name: {model_dir}")
    
    # Look for batch size
    batch_size = None
    for i, part in enumerate(parts):
        if part == "batch" and i + 1 < len(parts):
            batch_size = int(parts[i + 1])
            break
    
    if batch_size is None:
        raise ValueError(f"Could not extract batch size from folder name: {model_dir}")
    
    # Check for PRS and warmup
    prs_enabled = "PRS" in model_dir
    warmup_epochs = None
    
    for i, part in enumerate(parts):
        if part == "warmup" and i + 1 < len(parts):
            warmup_epochs = int(parts[i + 1])
            break
    
    # Return all the extracted information
    return model_name, dataset_name, batch_size, prs_enabled, warmup_epochs, epoch_num

# Custom function to load model directly from complete path
def load_model_from_folder(folder_path, device="cuda"):
    """
    Load model directly from the provided folder path.
    Handles both direct model directories and epoch subdirectories.
    
    Args:
        folder_path (str): Path to folder containing model.pth
        device (str): Device to load model on
    
    Returns:
        torch.nn.Module: Loaded model
    """
    folder_path = Path(folder_path)
    model_path = folder_path / "model.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Extract model info to determine architecture
    model_name, dataset_name, batch_size, prs_enabled, warmup_epochs, epoch_num = extract_model_info(folder_path)
    
    # Initialize model architecture
    input_channels = 3  # Default for CIFAR10
    if dataset_name == "MNIST":
        input_channels = 1
    
    model = get_model(model_name, input_channels=input_channels)
    
    # Load the weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Construct an informative model description
    model_desc = f"{model_name} ({dataset_name}, batch={batch_size}"
    if warmup_epochs is not None:
        model_desc += f", warmup={warmup_epochs}"
    if prs_enabled:
        model_desc += ", PRS"
    if epoch_num is not None:
        model_desc += f", epoch={epoch_num}"
    model_desc += ")"
    
    print(f"Loaded trained model from {model_path}")
    return model, model_desc

def main():
    """Main function to evaluate adversarial robustness of trained models."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness of trained models.")
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first model folder.")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the second model folder.")
    args = parser.parse_args()

    device = config["device"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load trained models directly from the provided paths
    model1, model1_desc = load_model_from_folder(args.folder1, device)
    model2, model2_desc = load_model_from_folder(args.folder2, device)
    
    # Extract dataset name from first model for consistency
    _, dataset_name, _, _, _, _ = extract_model_info(args.folder1)
    
    models_to_evaluate = {
        model1_desc: model1,
        model2_desc: model2,
    }
    
    print(f"Evaluating models:\n1. {model1_desc}\n2. {model2_desc}")

    # Load Dataset Based on the extracted dataset name
    train_dataset, test_dataset, _ = get_datasets(dataset_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Dictionary to store results
    results = {}

    # Run attacks on each model for each epsilon
    for model_name, model in models_to_evaluate.items():
        results[model_name] = {}

        for attack_name, attack_data in ATTACKS.items():
            results[model_name][attack_name] = {"Train": {}, "Test": {}}

            for epsilon in EPSILONS:
                train_acc = evaluate_robustness(model, train_loader, attack_data["attack_fn"], attack_data["params"], epsilon, device)
                test_acc = evaluate_robustness(model, test_loader, attack_data["attack_fn"], attack_data["params"], epsilon, device)

                results[model_name][attack_name]["Train"][str(epsilon)] = train_acc
                results[model_name][attack_name]["Test"][str(epsilon)] = test_acc

                print(f"{attack_name} Train Acc ({model_name}) at ε={epsilon}: {train_acc:.2f}%")
                print(f"{attack_name} Test Acc ({model_name}) at ε={epsilon}: {test_acc:.2f}%")

    # Create timestamp for unique filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to JSON
    os.makedirs(config["results_save_path"], exist_ok=True)
    results_path = os.path.join(config["results_save_path"], f"adversarial_results_{timestamp}_.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Adversarial robustness evaluation completed! Results saved to {results_path}")

if __name__ == "__main__":
    main()