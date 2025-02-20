import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  

# Import utility functions
from models import get_model
from utils import get_datasets, evaluate, compute_unique_activations, register_activation_hook, compute_major_regions, save_major_regions
from utils.logger import setup_logger  
from config import config

# ---------------------- Step 1: Set Seed for Reproducibility ----------------------
def set_seed(seed):
    """Ensure reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------- Step 2: Xavier Initialization with Seed ----------------------
def initialize_weights(model, seed):
    """Apply Xavier initialization to layers with a fixed seed for reproducibility."""
    torch.manual_seed(seed)  # Ensures reproducibility in initialization
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# ---------------------- Step 3: Decorator for NaN Checking in Loss & Gradients ----------------------
def check_nan(func):
    """Decorator to detect NaNs in loss or gradients and handle them."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            return None
        if torch.isnan(result).any():
            logger = kwargs.get('logger', None)
            if logger:
                logger.warning(f"NaN detected in {func.__name__}, skipping batch...")
            return None
        return result
    return wrapper

@check_nan
def compute_loss(criterion, outputs, labels, logger=None):
    """Compute loss and check for NaNs."""
    return criterion(outputs, labels)

@check_nan
def backward_pass(loss, optimizer, scaler, logger=None):
    """Perform backward pass and check for NaNs in gradients."""
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=5.0)
    return loss

# ---------------------- Step 4: Save Model and Checkpoint ----------------------
def save_model_checkpoint(model, optimizer, modelname, dataset_name, batch_size, metrics, logger):
    """Save trained model, optimizer state, and metadata."""
    save_dir = os.path.join("models", "saved", f"{modelname}_{dataset_name}_batch_{batch_size}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Model and metadata saved in {save_dir}")

# ---------------------- Step 5: Training Pipeline ----------------------
def train():
    """Training loop over different datasets and batch sizes."""
    set_seed(config["seed"])
    results = {}

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):

                # Initialize dynamic logger for this setting
                logger = setup_logger(modelname, dataset_name, batch_size)
                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size}")

                # Load dataset
                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, 
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                # Model initialization with Xavier and fixed seed
                model = get_model(modelname, input_channels).to(config["device"])
                initialize_weights(model, config["seed"])

                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
                criterion = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler("cuda")

                activations = {"penultimate": [], "skip_batch": False}
                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}                
                final_epoch_activations, final_epoch_labels = None, None

                for epoch in tqdm(range(config["epochs"]), desc=f"Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []

                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Model: {modelname} | Dataset: {dataset_name}")

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        activations["skip_batch"] = False
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                        optimizer.zero_grad()
                        
                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            loss = compute_loss(criterion, outputs, labels, logger)
                            if loss is None:
                                continue  

                        loss = backward_pass(loss, optimizer, scaler, logger)
                        if loss is None:
                            continue  

                        scaler.step(optimizer)
                        scaler.update()

                        if activations["skip_batch"]:
                            logger.warning(f"Skipping NaN batch | Epoch: {epoch+1} | Batch: {batch_idx}")
                            continue  
                        
                        epoch_loss += loss.item()
                        batch_labels.append(labels.cpu().numpy())

                        _, predicted = outputs.max(1)
                        correct_train += (predicted == labels).sum().item()
                        total_train += labels.size(0)
                    
                    # Convert activations to float32 for stability
                    if len(activations["penultimate"]) > 0:
                        epoch_activations = np.concatenate([act.astype(np.float32) for act in activations["penultimate"]], axis=0)
                        epoch_labels = np.concatenate(batch_labels, axis=0)
                    else:
                        logger.error(f"No activations collected in epoch {epoch+1}")
                        continue
                    
                    # Store only the last epoch activations for MR computation
                    if epoch == config["epochs"] - 1:
                        final_epoch_activations = epoch_activations
                        final_epoch_labels = epoch_labels

                    # Compute PRS Ratio
                    prs_ratio = compute_unique_activations(epoch_activations, logger) / len(train_dataset)
                    metrics["prs_ratios"].append(prs_ratio)

                    # Evaluate on Test Set
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    train_accuracy = 100 * correct_train / total_train
                    
                    # Store Metrics
                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}%")

                hook_handle.remove()

                # Compute & Save Major Regions (MR) using last epoch activations
                if final_epoch_activations is not None and final_epoch_labels is not None:
                    major_regions, unique_patterns = compute_major_regions(final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger)
                    save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, modelname, logger)
                else:
                    logger.error("Error: No valid final epoch activations found for MR computation!")

                results[f"{dataset_name}_batch_{batch_size}"] = metrics
                with open(os.path.join(config["results_save_path"], f"metrics_{dataset_name}_batch_{batch_size}.json"), "w") as f:
                    json.dump(metrics, f, indent=4)

                save_model_checkpoint(model, optimizer, modelname, dataset_name, batch_size, metrics, logger)

    logger.info("Training Complete")

if __name__ == "__main__":
    train()
