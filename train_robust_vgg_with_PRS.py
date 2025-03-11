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
    save_model_checkpoint, set_seed, freeze_final_layer, initialize_weights
)
from utils.logger import setup_logger
from utils.regularization import compute_mrv_loss, compute_hamming_loss
from config import config

def train():
    """Enhanced training loop with warm-up, PRS regularization, and stability features."""
    set_seed(config["seed"])
    results = {}

    # Enhanced training hyperparameters
    base_lr = 5e-4  # Reduced learning rate for stability
    weight_decay = 1e-4  # Weight decay for regularization
    gradient_accumulation_steps = 4  # Accumulate gradients for stability
    
    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                # Initialize dynamic logger for this setting
                logger = setup_logger(modelname, dataset_name, batch_size)
                warmup_epochs = config["warmup_epochs"]
                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size} | Warmup: {warmup_epochs}")
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

                metrics = {
                    "epoch": [], 
                    "train_accuracy": [], 
                    "test_accuracy": [], 
                    "prs_ratios": [], 
                    "learning_rates": [],
                    "mrv_loss": [],
                    "hamming_loss": []
                }
                
                activations = {"penultimate": [], "skip_batch": False}
                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)
                
                # ===========================
                # 🔹 WARM-UP STAGE
                # ===========================
                logger.info(f"Starting warm-up phase: {warmup_epochs} epochs")
                
                for epoch in tqdm(range(warmup_epochs), desc=f"Warm-up Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []
                    
                    current_lr = scheduler.get_last_lr()[0]
                    metrics["learning_rates"].append(current_lr)
                    logger.info(f"Warm-up Epoch {epoch+1}/{warmup_epochs} | Model: {modelname} | Dataset: {dataset_name} | LR: {current_lr:.6f}")

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
                            logger.warning(f"Skipping NaN batch | Warm-up Epoch: {epoch+1} | Batch: {batch_idx}")
                            continue  # Skip problematic batch

                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        
                        # Accumulate gradients over multiple batches
                        accumulated_batches += 1
                        
                        if accumulated_batches == gradient_accumulation_steps or batch_idx == len(train_loader) - 1:
                            # Check for NaN gradients before optimizer step
                            if any(torch.isnan(param.grad).any() for param in model.parameters() if param.grad is not None):
                                logger.warning(f"NaN detected in gradients. Skipping update | Warm-up Epoch: {epoch+1} | Batch: {batch_idx}")
                                # Reset gradients without applying them
                                optimizer.zero_grad()
                                accumulated_batches = 0
                                continue
                            
                            # Apply gradient clipping
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
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
                    
                    # Process warm-up activations
                    if activations["penultimate"]:  # Check if list is not empty
                        final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                        final_epoch_labels = np.concatenate(batch_labels, axis=0)

                        # Compute PRS Ratio
                        prs_ratio = compute_unique_activations(final_epoch_activations, logger) / train_dataset_size
                        metrics["prs_ratios"].append(prs_ratio)
                        
                        # Add placeholders for losses not used in warm-up
                        metrics["mrv_loss"].append(0.0)
                        metrics["hamming_loss"].append(0.0)
                    else:
                        logger.warning(f"No activations collected in warm-up epoch {epoch+1}")
                        metrics["prs_ratios"].append(0.0)
                        metrics["mrv_loss"].append(0.0)
                        metrics["hamming_loss"].append(0.0)
                        final_epoch_activations = None
                        final_epoch_labels = None

                    # Evaluate on Test Set during warm-up
                    hook_handle.remove()  # Remove hook during evaluation
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)
                    
                    train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
                    
                    # Store Metrics
                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    
                    logger.info(f"Warm-up Epoch {epoch+1}/{warmup_epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | PRS Ratio: {prs_ratio:.4f}")

                # Compute Major Regions after warm-up stage
                if final_epoch_activations is not None and final_epoch_labels is not None:
                    logger.info("Computing major regions after warm-up phase")
                    major_regions, unique_patterns = compute_major_regions(
                        final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger
                    )
                else:
                    logger.error("Unable to compute major regions: no activations available after warm-up")
                    continue  # Skip to next configuration if we can't compute major regions
                
                # ===========================
                # 🔹 FREEZE FINAL LAYER
                # ===========================
                freeze_final_layer(model, modelname, logger)
                
                # Recreate optimizer and scheduler for PRS stage
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=base_lr, 
                    weight_decay=weight_decay
                )
                
                scheduler = CosineAnnealingLR(
                    optimizer, 
                    T_max=config["epochs"] - warmup_epochs, 
                    eta_min=base_lr / 10
                )
                
                # ===========================
                # 🔹 PRS REGULARIZATION STAGE
                # ===========================
                logger.info(f"Starting PRS regularization phase: {config['epochs'] - warmup_epochs} epochs")
                
                for epoch in tqdm(range(warmup_epochs, config["epochs"]), desc=f"PRS Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    epoch_mrv_loss, epoch_hamming_loss = 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []
                    
                    current_lr = scheduler.get_last_lr()[0]
                    metrics["learning_rates"].append(current_lr)
                    logger.info(f"PRS Epoch {epoch+1}/{config['epochs']} | Model: {modelname} | Dataset: {dataset_name} | LR: {current_lr:.6f}")

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
                            
                            # Add PRS regularization losses
                            if len(activations["penultimate"]) > 0:
                                batch_activations = torch.cat(activations["penultimate"], dim=0)
                                
                                # Compute MRV and Hamming losses
                                mrv_loss = compute_mrv_loss(batch_activations, labels, major_regions)
                                hamming_loss = compute_hamming_loss(batch_activations, labels, major_regions)
                                
                                # Scale and add regularization losses
                                # Check if the losses are already tensor objects or scalars
                                if isinstance(mrv_loss, torch.Tensor):
                                    scaled_mrv_loss = config["lambda_mrv"] * mrv_loss / gradient_accumulation_steps
                                    epoch_mrv_loss += scaled_mrv_loss.item() * gradient_accumulation_steps
                                else:
                                    scaled_mrv_loss = config["lambda_mrv"] * mrv_loss / gradient_accumulation_steps
                                    epoch_mrv_loss += scaled_mrv_loss * gradient_accumulation_steps
                                
                                if isinstance(hamming_loss, torch.Tensor):
                                    scaled_hamming_loss = config["lambda_hamming"] * hamming_loss / gradient_accumulation_steps
                                    epoch_hamming_loss += scaled_hamming_loss.item() * gradient_accumulation_steps
                                else:
                                    scaled_hamming_loss = config["lambda_hamming"] * hamming_loss / gradient_accumulation_steps
                                    epoch_hamming_loss += scaled_hamming_loss * gradient_accumulation_steps
                                
                                # Add regularization to main loss
                                loss += scaled_mrv_loss + scaled_hamming_loss

                        if activations["skip_batch"]:
                            logger.warning(f"Skipping NaN batch | PRS Epoch: {epoch+1} | Batch: {batch_idx}")
                            continue  # Skip problematic batch

                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        
                        # Accumulate gradients over multiple batches
                        accumulated_batches += 1
                        
                        if accumulated_batches == gradient_accumulation_steps or batch_idx == len(train_loader) - 1:
                            # Check for NaN gradients before optimizer step
                            if any(torch.isnan(param.grad).any() for param in model.parameters() if param.grad is not None):
                                logger.warning(f"NaN detected in gradients. Skipping update | PRS Epoch: {epoch+1} | Batch: {batch_idx}")
                                # Reset gradients without applying them
                                optimizer.zero_grad()
                                accumulated_batches = 0
                                continue
                            
                            # Apply gradient clipping
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
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
                    
                    # Process PRS stage activations
                    if activations["penultimate"]:  # Check if list is not empty
                        final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                        final_epoch_labels = np.concatenate(batch_labels, axis=0)

                        # Compute PRS Ratio
                        prs_ratio = compute_unique_activations(final_epoch_activations, logger) / train_dataset_size
                        metrics["prs_ratios"].append(prs_ratio)
                        
                        # Record regularization losses
                        metrics["mrv_loss"].append(epoch_mrv_loss)
                        metrics["hamming_loss"].append(epoch_hamming_loss)
                    else:
                        logger.warning(f"No activations collected in PRS epoch {epoch+1}")
                        metrics["prs_ratios"].append(0.0)
                        metrics["mrv_loss"].append(0.0)
                        metrics["hamming_loss"].append(0.0)
                        final_epoch_activations = None
                        final_epoch_labels = None

                    # Evaluate on Test Set during PRS stage
                    hook_handle.remove()  # Remove hook during evaluation
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)
                    
                    train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
                    
                    # Store Metrics
                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    
                    logger.info(f"PRS Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | PRS Ratio: {prs_ratio:.4f} | MRV Loss: {epoch_mrv_loss:.4f} | Hamming Loss: {epoch_hamming_loss:.4f}")

                # Remove hook after training
                hook_handle.remove()

                # Compute and Save MR/ER using final epoch activations
                if final_epoch_activations is not None and final_epoch_labels is not None:
                    logger.info("Computing final major regions after PRS phase")
                    major_regions, unique_patterns = compute_major_regions(
                        final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger
                    )
                    save_major_regions(
                        major_regions, unique_patterns, dataset_name, batch_size, 
                        modelname, logger, prs_enabled=True, warmup_epochs=warmup_epochs
                    )
                else:
                    logger.warning("Unable to compute final major regions: no activations available")

                # Ensure results directory exists
                results_save_path = config["results_save_path"]
                os.makedirs(results_save_path, exist_ok=True)

                # Save final results
                results[f"{dataset_name}_batch_{batch_size}"] = metrics

                # Save final model checkpoint
                
                save_model_checkpoint(
                    model, optimizer, modelname, dataset_name, batch_size, 
                    metrics, logger, scheduler=scheduler, prs_enabled=True, 
                    config=config
                )

    logger.info("Training Complete")


if __name__ == "__main__":
    train()