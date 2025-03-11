import sys
import os

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

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
    Extract model parameters from folder path.
    Example path: /home/tair/project_root/models/saved/VGG16_CIFAR10_batch_128_warmup_50_PRS
    """
    basename = os.path.basename(folder_path)
    parts = basename.split('_')
    
    # Extract basic information
    model_name = parts[0]  # e.g., 'VGG16'
    dataset_name = parts[1]  # e.g., 'CIFAR10'
    
    # Look for batch size
    batch_size = None
    for i, part in enumerate(parts):
        if part == "batch" and i + 1 < len(parts):
            batch_size = int(parts[i + 1])
            break
    
    if batch_size is None:
        raise ValueError(f"Could not extract batch size from folder name: {basename}")
    
    # Check for PRS and warmup
    prs_enabled = "PRS" in basename
    warmup_epochs = 50  # default
    
    for i, part in enumerate(parts):
        if part == "warmup" and i + 1 < len(parts):
            warmup_epochs = int(parts[i + 1])
            break
    
    return model_name, dataset_name, batch_size, prs_enabled, warmup_epochs

# Custom function to load model directly from complete path
def load_model_from_folder(folder_path, device="cuda"):
    """
    Load model directly from the provided folder path.
    
    Args:
        folder_path (str): Path to folder containing model.pth
        device (str): Device to load model on
    
    Returns:
        torch.nn.Module: Loaded model
    """
    model_path = os.path.join(folder_path, "model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Extract model info to determine architecture
    model_name, _, _, _, _ = extract_model_info(folder_path)
    
    # Initialize model architecture
    model = get_model(model_name, input_channels=3)
    
    # Load the weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loaded trained model from {model_path}")
    return model

def main():
    """Main function to evaluate adversarial robustness of trained models."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness of trained models.")
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first model folder.")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the second model folder.")
    args = parser.parse_args()

    device = config["device"] if torch.cuda.is_available() else "cpu"
    
    # Extract model parameters from folder paths (for debugging/info)
    model1_info = extract_model_info(args.folder1)
    model2_info = extract_model_info(args.folder2)
    
    print(f"Model 1 info: {model1_info}")
    print(f"Model 2 info: {model2_info}")

    # Load trained models directly from the provided paths
    models_to_evaluate = {
        "Model 1": load_model_from_folder(args.folder1, device),
        "Model 2": load_model_from_folder(args.folder2, device),
    }

    # Load CIFAR-10 Training & Test Dataset
    train_dataset, test_dataset, _ = get_datasets("CIFAR10")
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

    # Save results to JSON
    os.makedirs(config["results_save_path"], exist_ok=True)
    results_path = os.path.join(config["results_save_path"], "adversarial_results_cnn.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Adversarial robustness evaluation completed! Results saved to {results_path}")

if __name__ == "__main__":
    main()