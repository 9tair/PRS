import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from config import config

from utils import fgsm_attack, bim_attack, pgd_attack, get_datasets
from models import load_trained_model

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


def main():
    """Main function to evaluate adversarial robustness of trained models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Trained Models
    models_to_evaluate = {
        "Simple (Batch_128)": load_trained_model("CNN-6", "CIFAR10", batch_size=128, device=device, prs_enabled=False, warmup_epochs=config["warmup_epochs"]),
        "PRS (Batch_128)": load_trained_model("CNN-6", "CIFAR10", batch_size=128, device=device, prs_enabled=True, warmup_epochs=config["warmup_epochs"]),
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

                results[model_name][attack_name]["Train"][epsilon] = train_acc
                results[model_name][attack_name]["Test"][epsilon] = test_acc

                print(f"{attack_name} Train Acc ({model_name}) at ε={epsilon}: {train_acc:.2f}%")
                print(f"{attack_name} Test Acc ({model_name}) at ε={epsilon}: {test_acc:.2f}%")

    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    results_path = "results/adversarial_results_new.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Adversarial robustness evaluation completed! Results saved to {results_path}")

if __name__ == "__main__":
    main()
