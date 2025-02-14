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
from models import CustomCNN, get_model
from utils import get_datasets, evaluate, compute_unique_activations, register_activation_hook
from config import config

# Function to set the seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    """Training loop over different datasets and batch sizes"""
    set_seed(config["seed"])
    results = {}

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
            model = get_model(config["model"], input_channels).to(config["device"])
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
            criterion = nn.CrossEntropyLoss()

            # Mixed Precision Training
            scaler = torch.amp.GradScaler("cuda")

            metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
            epoch_activations = {}

            # Register Hook
            activations = {"penultimate": []}
            hook_handle = register_activation_hook(model, activations)  # Hook into penultimate layer

            for epoch in tqdm(range(config["epochs"]), desc=f"Training {dataset_name} | Batch {batch_size}"):
                model.train()
                epoch_loss = 0
                correct_train = 0
                total_train = 0
                activations["penultimate"].clear()

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                    optimizer.zero_grad()
                    
                    # Enable Mixed Precision
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()

                    # 🚀 **Apply Gradient Clipping**
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()

                    # Calculate training accuracy
                    _, predicted = outputs.max(1)
                    correct_train += (predicted == labels).sum().item()
                    total_train += labels.size(0)

                # Collect epoch activations
                epoch_activations[f"epoch_{epoch+1}"] = np.concatenate(activations["penultimate"], axis=0)

                # Compute PRS Ratio
                prs_ratio = compute_unique_activations(epoch_activations[f"epoch_{epoch+1}"]) / len(train_dataset)
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

            results[f"{dataset_name}_batch_{batch_size}"] = metrics
            metrics_path = os.path.join(config["results_save_path"], f"metrics_{dataset_name}_batch_{batch_size}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

if __name__ == "__main__":
    train()
