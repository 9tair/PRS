import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  

# Import utility functions
from models import get_model
from utils import (
    get_datasets, evaluate, compute_unique_activations, 
    register_activation_hook, compute_major_regions, save_major_regions,
    save_model_checkpoint, set_seed
)
from utils.logger import setup_logger  
from config import config

def train():
    """Main training loop handling different datasets and batch sizes."""
    set_seed(config["seed"])
    results = {}

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                # Initialize logger for this model, dataset, and batch size
                logger = setup_logger(modelname, dataset_name, batch_size)
                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size}")

                # Load dataset
                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                train_dataset_size = len(train_dataset)

                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, 
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                # Initialize model, optimizer, and loss function
                model = get_model(modelname, input_channels).to(config["device"])
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
                criterion = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler("cuda")

                # Metrics storage
                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
                activations = {"penultimate": [], "skip_batch": False}

                # Register activation hooks
                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                # Define checkpointing epochs
                total_epochs = config["epochs"]
                save_epochs = set(range(50, total_epochs + 1, 50))  # Every 50 epochs
                save_epochs.add(total_epochs)  # Always save the last epoch

                for epoch in tqdm(range(1, total_epochs + 1), desc=f"Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []

                    logger.info(f"Epoch {epoch}/{total_epochs} | Model: {modelname} | Dataset: {dataset_name}")

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        activations["skip_batch"] = False

                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                        optimizer.zero_grad()
                        
                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        if activations["skip_batch"]:
                            logger.warning(f"Skipping NaN batch | Epoch: {epoch} | Batch: {batch_idx}")
                            continue

                        scaler.scale(loss).backward()

                        if any(torch.isnan(param.grad).any() for param in model.parameters()):
                            logger.warning(f"NaN detected in gradients. Skipping update | Epoch: {epoch} | Batch: {batch_idx}")
                            continue

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                        scaler.step(optimizer)
                        scaler.update()
                        
                        epoch_loss += loss.item()
                        batch_labels.append(labels.cpu().numpy())

                        _, predicted = outputs.max(1)
                        correct_train += (predicted == labels).sum().item()
                        total_train += labels.size(0)
                    
                    # Compute PRS Ratio
                    final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                    final_epoch_labels = np.concatenate(batch_labels, axis=0)
                    prs_ratio = compute_unique_activations(final_epoch_activations, logger) / train_dataset_size
                    metrics["prs_ratios"].append(prs_ratio)

                    # Evaluate on Test Set
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    train_accuracy = 100 * correct_train / total_train
                    
                    # Store Metrics
                    metrics["epoch"].append(epoch)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)

                    logger.info(f"Epoch {epoch}/{total_epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")

                    # Save checkpoint, compute major regions, and save every 50 epochs or last epoch
                    if epoch in save_epochs:
                        epoch_dir = os.path.join("models", "saved", modelname, dataset_name, f"batch_{batch_size}", f"epoch_{epoch}")
                        os.makedirs(epoch_dir, exist_ok=True)

                        # Compute and Save Major Regions & Unique Patterns
                        major_regions, unique_patterns = compute_major_regions(final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger)
                        save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, modelname, logger, warmup_epochs=config["warmup_epochs"])

                        # Save model checkpoint
                        save_model_checkpoint(
                            model, optimizer, modelname, dataset_name, batch_size, 
                            metrics, logger, extra_tag=f"epoch_{epoch}"
                        )

                        # Save model weights separately
                        torch.save(model.state_dict(), os.path.join(epoch_dir, "weights.pth"))

                hook_handle.remove()  # Remove hooks after training

                # Store final results
                results[f"{dataset_name}_batch_{batch_size}"] = metrics

    logger.info("Training Complete")

if __name__ == "__main__":
    train()
