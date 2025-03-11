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
    gradient_accumulation_steps = 4  # Accumulate gradients over multiple batches
    
    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                # Initialize dynamic logger for this setting
                logger = setup_logger(modelname, dataset_name, batch_size)
                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size}")
                logger.info(f"Using lr={base_lr}, weight_decay={weight_decay}, grad_accum_steps={gradient_accumulation_steps}")

                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                train_dataset_size = len(train_dataset)
                effective_batch_size = batch_size * gradient_accumulation_steps
                
                logger.info(f"Actual batch size: {batch_size}, Effective batch size with accumulation: {effective_batch_size}")

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, 
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                # Get model and apply improved initialization
                model = get_model(modelname, input_channels).to(config["device"])
                model = initialize_weights(model)
                
                # Setup optimizer with weight decay
                optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
                
                # Learning rate scheduler
                scheduler = CosineAnnealingLR(
                    optimizer, 
                    T_max=config["epochs"], 
                    eta_min=base_lr / 10
                )
                
                criterion = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler("cuda")

                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": [], "learning_rates": []}
                activations = {"penultimate": [], "skip_batch": False}

                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                for epoch in tqdm(range(config["epochs"]), desc=f"Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []
                    
                    current_lr = scheduler.get_last_lr()[0]
                    metrics["learning_rates"].append(current_lr)
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Model: {modelname} | Dataset: {dataset_name} | LR: {current_lr:.6f}")

                    # Reset gradients at the start of each epoch
                    optimizer.zero_grad()
                    accumulated_batches = 0

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        activations["skip_batch"] = False
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                        
                        # Forward pass with mixed precision
                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels) / gradient_accumulation_steps  # Scale loss

                        if activations["skip_batch"]:
                            logger.warning(f"Skipping NaN batch | Epoch: {epoch+1} | Batch: {batch_idx}")
                            continue  # Skip problematic batch

                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        
                        # Accumulate gradients over multiple batches
                        accumulated_batches += 1
                        
                        if accumulated_batches == gradient_accumulation_steps or batch_idx == len(train_loader) - 1:
                            # Check for NaN gradients before optimizer step
                            if any(torch.isnan(param.grad).any() for param in model.parameters() if param.grad is not None):
                                logger.warning(f"NaN detected in gradients. Skipping update | Epoch: {epoch+1} | Batch: {batch_idx}")
                                # Reset gradients without applying them
                                optimizer.zero_grad()
                                accumulated_batches = 0
                                continue
                            
                            # Apply gradient clipping
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Reduced from 5.0
                            
                            # Step optimizer and update scaler
                            scaler.step(optimizer)
                            scaler.update()
                            
                            # Reset gradients
                            optimizer.zero_grad()
                            accumulated_batches = 0

                        epoch_loss += loss.item() * gradient_accumulation_steps  # Re-scale loss for logging
                        batch_labels.append(labels.cpu().numpy())

                        _, predicted = outputs.max(1)
                        correct_train += (predicted == labels).sum().item()
                        total_train += labels.size(0)
                    
                    # Update learning rate scheduler at the end of each epoch
                    scheduler.step()
                    
                    # Process activations
                    if activations["penultimate"]:  # Check if list is not empty
                        final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                        final_epoch_labels = np.concatenate(batch_labels, axis=0)

                        # Compute PRS Ratio
                        prs_ratio = compute_unique_activations(final_epoch_activations, logger) / train_dataset_size
                        metrics["prs_ratios"].append(prs_ratio)
                    else:
                        logger.warning(f"No activations collected in epoch {epoch+1}")
                        metrics["prs_ratios"].append(0.0)
                        final_epoch_activations = None
                        final_epoch_labels = None

                    # Evaluate on Test Set
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
                    
                    # Store Metrics
                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | LR: {current_lr:.6f}")

                hook_handle.remove()  # Ensure hook is removed after training

                # Compute and Save MR/ER using only LAST epoch activations
                if final_epoch_activations is not None and final_epoch_labels is not None:
                    major_regions, unique_patterns = compute_major_regions(
                        final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger
                    )
                    save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, modelname, logger, nan_enabled=True, warmup_epochs=config["warmup_epochs"])
                else:
                    logger.warning("Unable to compute major regions: no activations available")

                # Ensure results directory exists
                results_save_path = config["results_save_path"]
                os.makedirs(results_save_path, exist_ok=True)

                results[f"{dataset_name}_batch_{batch_size}"] = metrics
                
                save_model_checkpoint(
                    model, optimizer, scheduler, modelname, dataset_name, batch_size, 
                    metrics, logger
                )

    logger.info("Training Complete")


if __name__ == "__main__":
    train()