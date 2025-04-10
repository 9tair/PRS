import os
import json
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
    save_model_checkpoint, set_seed, freeze_final_layer
)
from utils.logger import setup_logger  
from utils.regularization import compute_mrv_loss, compute_hamming_loss
from config import config

def load_checkpoint_if_exists(model, optimizer, modelname, dataset_name, batch_size, logger):
    """Loads a model checkpoint from `epoch_{warmup_epochs}` instead of the latest epoch."""
    
    checkpoint_path = os.path.join("models", "saved", f"{modelname}_{dataset_name}_batch_{batch_size}")
    warmup_epoch = config["warmup_epochs"]  # Get warm-up epoch from config
    epoch_path = os.path.join(checkpoint_path, f"epoch_{warmup_epoch}")  # Load from warmup epoch

    if os.path.exists(epoch_path):
        logger.info(f"Loading checkpoint from epoch {warmup_epoch}: {epoch_path}")

        try:
            model.load_state_dict(torch.load(os.path.join(epoch_path, "model.pth"), map_location=config["device"]))
            optimizer.load_state_dict(torch.load(os.path.join(epoch_path, "optimizer.pth"), map_location=config["device"]))

            scheduler_state = torch.load(os.path.join(epoch_path, "scheduler.pth"), map_location=config["device"]) if os.path.exists(os.path.join(epoch_path, "scheduler.pth")) else None
            
            with open(os.path.join(epoch_path, "major_regions.json"), "r") as f:
                major_regions = json.load(f)

            with open(os.path.join(epoch_path, "unique_patterns.json"), "r") as f:
                unique_patterns = json.load(f)

            model.to(config["device"])
            model.train()
            
            return True, model, optimizer, scheduler_state, major_regions, unique_patterns
        except Exception as e:
            logger.error(f"Error loading checkpoint from epoch {warmup_epoch}: {e}")
            return False, model, optimizer, None, None, None
    else:
        logger.warning(f"Checkpoint for epoch {warmup_epoch} not found at {epoch_path}. Starting from scratch.")
        return False, model, optimizer, None, None, None

def train():
    """Training loop with warm-up, final layer freezing, and PRS regularization."""
    set_seed(config["seed"])
    results = {}

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                logger = setup_logger(modelname, dataset_name, batch_size)
                warmup_epochs = config["warmup_epochs"]
                total_epochs = config["epochs"]
                save_epochs = set(range(warmup_epochs, total_epochs + 1, 10))  # Save every 10 epoch
                save_epochs.add(total_epochs)  # Ensure last epoch is saved

                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size} | Warmup: {warmup_epochs}")

                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, 
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                model = get_model(modelname, input_channels).to(config["device"])
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
                criterion = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler()        
                
                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
                activations = {
                    "penultimate": [],   
                    "skip_batch": False,
                    "current": None
                }
                
                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                # Load checkpoint **only once at the beginning**
                checkpoint_exists, model, optimizer, scheduler_state, major_regions, unique_patterns = load_checkpoint_if_exists(
                    model, optimizer, modelname, dataset_name, batch_size, logger
                )

                if scheduler_state:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
                    scheduler.load_state_dict(scheduler_state)
                else:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

                if checkpoint_exists:
                    logger.info("Resuming training after warm-up.")
                    start_epoch = warmup_epochs
                else:
                    logger.info("Starting warm-up training.")
                    start_epoch = 0
                    # ===========================
                    # WARM-UP STAGE
                    # ===========================
                    for epoch in tqdm(range(warmup_epochs), desc=f"Warm-up Training {dataset_name} | Batch {batch_size}"):
                        model.train()
                        epoch_loss, correct_train, total_train = 0, 0, 0
                        activations = {
                            "penultimate": [],   
                            "skip_batch": False,
                            "current": None
                        }

                        batch_labels = []

                        for batch_idx, (inputs, labels) in enumerate(train_loader):
                            activations["skip_batch"] = False
                            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                            optimizer.zero_grad()
                            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            
                            epoch_loss += loss.item()
                            
                            _, predicted = outputs.max(1)
                            correct_train += (predicted == labels).sum().item()
                            total_train += labels.size(0)

                            batch_labels.append(labels.cpu().numpy())

                        train_accuracy = 100 * correct_train / total_train
                        
                        if len(activations["penultimate"]) > 0:
                            final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                            final_epoch_labels = np.concatenate(batch_labels, axis=0)
                            prs_ratio = compute_unique_activations(final_epoch_activations, logger) / len(train_dataset)
                            metrics["prs_ratios"].append(prs_ratio)
                        else:
                            logger.warning(f"No activations captured in epoch {epoch+1}. Check activation hook.")
                            prs_ratio = 0
                            metrics["prs_ratios"].append(prs_ratio)
                        
                        hook_handle.remove()
                        test_accuracy = evaluate(model, test_loader, config["device"])
                        hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)
                        
                        metrics["epoch"].append(epoch + 1)
                        metrics["train_accuracy"].append(train_accuracy)
                        metrics["test_accuracy"].append(test_accuracy)
                        logger.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")
                        
                    # Check if we have collected activations during warm-up
                    if len(activations["penultimate"]) > 0:
                        # 🔹 Compute Major Regions before PRS Regularization
                        major_regions, unique_patterns = compute_major_regions(final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger)
                    else:
                        logger.error("No activations collected during warm-up. Cannot compute major regions.")
                        continue  # Skip to next configuration
                    
                # ===========================
                # FREEZE FINAL LAYER
                # ===========================
                freeze_final_layer(model, modelname, logger)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=0)

                for epoch in tqdm(range(start_epoch, total_epochs), desc=f"PRS Regularized Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0  # Added epoch_loss tracking

                    activations["penultimate"].clear()
                    batch_labels = []
                    
                    total_mrv_loss, total_hamming_loss = 0.0, 0.0

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        activations["current"] = None
                        activations["skip_batch"] = False
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                        optimizer.zero_grad()

                        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        if activations["current"] is not None:
                            try:
                                mrv_loss = compute_mrv_loss(activations["current"], labels, major_regions)
                                hamming_loss = compute_hamming_loss(activations["current"], labels, major_regions)

                                # Add losses to total
                                mrv_loss_value = mrv_loss.item() if hasattr(mrv_loss, 'item') else mrv_loss
                                hamming_loss_value = hamming_loss.item() if hasattr(hamming_loss, 'item') else hamming_loss

                                total_mrv_loss += mrv_loss_value
                                total_hamming_loss += hamming_loss_value

                                # Add to main loss
                                loss += config["lambda_mrv"] * mrv_loss + config["lambda_hamming"] * hamming_loss
                            except Exception as e:
                                logger.error(f"Error computing regularization loss: {e}")

                        # Store current batch's activations for PRS metrics
                        if activations["current"] is not None:
                            activations["penultimate"].append(activations["current"].detach().cpu())

                        epoch_loss += loss.item()  # Added loss tracking

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        _, predicted = outputs.max(1)
                        correct_train += (predicted == labels).sum().item()
                        total_train += labels.size(0)
                        
                        batch_labels.append(labels.cpu().numpy())
                        
                        
                    # Calculate train accuracy for this epoch
                    train_accuracy = 100 * correct_train / total_train  # Added missing calculation

                    if len(activations["penultimate"]) > 0:
                        final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                        final_epoch_labels = np.concatenate(batch_labels, axis=0)
                        prs_ratio = compute_unique_activations(final_epoch_activations, logger) / len(train_dataset)
                        metrics["prs_ratios"].append(prs_ratio)
                    else:
                        logger.warning(f"No activations captured in epoch {epoch+1}. Check activation hook.")
                        prs_ratio = 0
                        metrics["prs_ratios"].append(prs_ratio)

                    # 🔹 Prevent test images from influencing PRS calculation
                    hook_handle.remove()
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    
                    avg_mrv_loss = total_mrv_loss / len(train_loader)
                    avg_hamming_loss = total_hamming_loss / len(train_loader)
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | Avg MRV Loss: {avg_mrv_loss} | Avg Hamming Loss: {avg_hamming_loss}")

                    # Save checkpoint, compute major regions, and save every 50 epochs or last epoch
                    if epoch in save_epochs:
                        if len(activations["penultimate"]) > 0:
                            final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                            final_epoch_labels = np.concatenate(batch_labels, axis=0)

                            major_regions, unique_patterns = compute_major_regions(
                                final_epoch_activations, final_epoch_labels,
                                num_classes=10, logger=logger
                            )

                            save_model_checkpoint(
                                model, optimizer, modelname, dataset_name, batch_size,
                                metrics, logger, config=config,
                                extra_tag=None, epoch=epoch, prs_enabled=True,
                                major_regions=major_regions, unique_patterns=unique_patterns
                            )
                        else:
                            logger.warning(f"Skipping PRS region computation and checkpoint save at epoch {epoch} due to empty activations.")

                hook_handle.remove()  # Remove hooks after training

                results[f"{dataset_name}_batch_{batch_size}"] = metrics

    logger.info("Training Complete")

if __name__ == "__main__":
    train()
