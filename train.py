import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  

# Import refactored utility functions
from models import get_model
from utils import get_datasets, evaluate, compute_unique_activations, register_activation_hook
from utils import compute_major_regions, save_major_regions
from config import config

# Function to set the seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model_checkpoint(model, optimizer, dataset_name, batch_size, metrics):
    """Save trained model, optimizer state, and metadata."""
    save_dir = os.path.join("models", "saved", f"{dataset_name}_batch_{batch_size}")
    os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists

    # Save model
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # Save optimizer state
    optimizer_path = os.path.join(save_dir, "optimizer.pth")
    torch.save(optimizer.state_dict(), optimizer_path)

    # Save training configuration
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Save training metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model and metadata saved in {save_dir}")

def train():
    """Training loop over different datasets and batch sizes"""
    set_seed(config["seed"])
    results = {}

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                # Load dataset dynamically
                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)

                # Set data loaders
                generator = torch.Generator()
                generator.manual_seed(config["seed"])
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, 
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=generator
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                # Initialize model
                model = get_model(modelname, input_channels).to(config["device"])
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
                criterion = nn.CrossEntropyLoss()

                # Mixed Precision Training
                scaler = torch.amp.GradScaler("cuda")

                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
                epoch_activations = []
                epoch_labels = []

                # Register Hook
                activations = {"penultimate": []}
                hook_handle = register_activation_hook(model, activations)  # Hook into penultimate layer

                for epoch in tqdm(range(config["epochs"]), desc=f"Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss = 0
                    correct_train = 0
                    total_train = 0
                    activations["penultimate"].clear()
                    batch_labels = []

                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                        optimizer.zero_grad()
                        
                        # Enable Mixed Precision
                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # 🔹 Fix float16 issue: Convert activations to float32 immediately
                        activations["penultimate"] = [act.astype(np.float32) for act in activations["penultimate"]]

                        epoch_loss += loss.item()

                        # Store labels
                        batch_labels.append(labels.cpu().numpy())

                        # Calculate training accuracy
                        _, predicted = outputs.max(1)
                        correct_train += (predicted == labels).sum().item()
                        total_train += labels.size(0)

                    # Collect epoch activations & labels
                    epoch_activations.append(np.concatenate(activations["penultimate"], axis=0))
                    epoch_labels.append(np.concatenate(batch_labels, axis=0))

                    # Compute PRS Ratio
                    prs_ratio = compute_unique_activations(epoch_activations[-1]) / len(train_dataset)
                    metrics["prs_ratios"].append(prs_ratio)

                    # Evaluate on Test Set
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    train_accuracy = 100 * correct_train / total_train

                    # Store Metrics
                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)

                    tqdm.write(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | PRS Ratio: {prs_ratio:.4f}")

                hook_handle.remove()

                # Compute and Save MR/ER/MRV
                major_regions = compute_major_regions(np.vstack(epoch_activations), np.hstack(epoch_labels), num_classes=10)
                save_major_regions(major_regions, dataset_name, batch_size)

                results[f"{dataset_name}_batch_{batch_size}"] = metrics
                metrics_path = os.path.join(config["results_save_path"], f"metrics1_{dataset_name}_batch_{batch_size}.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f)

                # 🔹 Save model, optimizer, and training metadata
                save_model_checkpoint(model, optimizer, f'{modelname}_{dataset_name}', batch_size, metrics)

if __name__ == "__main__":
    train()
