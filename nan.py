import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import utility functions
from models import get_model
from utils import (
    get_datasets, evaluate, compute_unique_activations, 
    register_activation_hook, compute_major_regions, save_major_regions,
    save_model_checkpoint, set_seed, initialize_weights
)
from utils.logger import setup_logger
from config import config

def train():
    """Main training loop with enhanced stability features."""
    set_seed(config["seed"])
    results = {}

    # Training hyperparameters
    base_lr = 5e-4  # Reduced from 1e-3
    weight_decay = 1e-4  # Added weight decay
    
    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                # Initialize dynamic logger for this setting
                logger = setup_logger(modelname, dataset_name, batch_size)
                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size}")
                logger.info(f"Using lr={base_lr}, weight_decay={weight_decay}")

                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, 
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                # Get model and apply improved initialization
                model = get_model(modelname, input_channels).to(config["device"])
                initialize_weights(model)  # Assume this doesn't return anything
                
                # Setup optimizer with weight decay
                optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
                
                # Learning rate scheduler
                scheduler = CosineAnnealingLR(
                    optimizer, 
                    T_max=config["epochs"], 
                    eta_min=base_lr / 10
                )
                
                criterion = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler()  # Fixed: no argument needed

                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": [], "learning_rates": []}
                activations = {"penultimate": [], "skip_batch": False}

                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)
                
                total_epochs = config["epochs"]
                save_epochs = set(range(50, total_epochs + 1, 50)) 
                save_epochs.add(total_epochs)  

                for epoch in tqdm(range(config["epochs"]), desc=f"Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"] = []  # Clear the list for this epoch
                    batch_labels = []
                    
                    logger.info(f"Epoch {epoch+1}/{total_epochs} | Model: {modelname} | Dataset: {dataset_name}")

                    current_lr = scheduler.get_last_lr()[0]
                    metrics["learning_rates"].append(current_lr)
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Model: {modelname} | Dataset: {dataset_name} | LR: {current_lr:.6f}")

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        activations["skip_batch"] = False
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                        
                        # Forward pass with mixed precision
                        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):  # Fixed: proper autocast syntax
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        # Check if the hook flagged this batch to be skipped (e.g., due to NaN)
                        if activations["skip_batch"]:
                            logger.warning(f"Skipping problematic batch | Epoch: {epoch+1} | Batch: {batch_idx}")
                            continue

                        # Backward pass with gradient scaling
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        
                        # Apply gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Step optimizer and update scaler
                        scaler.step(optimizer)
                        scaler.update()

                        epoch_loss += loss.item()
                        batch_labels.append(labels.cpu().numpy())

                        _, predicted = outputs.max(1)
                        correct_train += (predicted == labels).sum().item()
                        total_train += labels.size(0)
                    
                    # Update learning rate scheduler at the end of each epoch
                    scheduler.step()
                    
                    # Calculate PRS ratio if we have activations
                    if len(activations["penultimate"]) > 0:
                        final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                        final_epoch_labels = np.concatenate(batch_labels, axis=0)
                        prs_ratio = compute_unique_activations(final_epoch_activations, logger) / len(train_dataset)
                        metrics["prs_ratios"].append(prs_ratio)
                        logger.info(f"Epoch {epoch+1} | PRS Ratio: {prs_ratio:.4f}")
                    else:
                        logger.warning(f"No activations captured in epoch {epoch+1}. Check activation hook.")
                        metrics["prs_ratios"].append(0)
                    
                    # Evaluate on Test Set
                    hook_handle.remove()  # Temporarily remove hook during evaluation
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)  # Re-register hook
                    
                    train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
                    
                    # Store Metrics
                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | LR: {current_lr:.6f}")

                    # Save checkpoint at specified epochs
                    if epoch + 1 in save_epochs:
                        # Compute major regions using the actual activations and labels
                        if len(activations["penultimate"]) > 0:
                            major_regions, unique_patterns = compute_major_regions(
                                final_epoch_activations, 
                                final_epoch_labels, 
                                num_classes=10, 
                                logger=logger
                            )
                            
                            # Save the model checkpoint with the computed regions
                            save_model_checkpoint(
                                model, optimizer, modelname, dataset_name, batch_size, 
                                metrics, logger, config=config, 
                                epoch=epoch + 1,
                                major_regions=major_regions, 
                                unique_patterns=unique_patterns,
                                extra_tag="nan"
                            )
                        else:
                            logger.error(f"Cannot compute major regions for epoch {epoch+1} - no activations collected")
                        
                # Make sure to remove the hook after training
                hook_handle.remove()
                
                # Store results for this configuration
                results[f"{modelname}_{dataset_name}_batch_{batch_size}"] = metrics

    logger.info("Training Complete")
    return results

if __name__ == "__main__":
    train()