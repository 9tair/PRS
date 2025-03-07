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
from utils import (
    get_datasets, evaluate, compute_unique_activations, 
    register_activation_hook, compute_major_regions, save_major_regions
)
from utils.logger import setup_logger  
from utils.regularization import compute_mrv_loss, compute_hamming_loss
from config import config

def set_seed(seed):
    """Ensure reproducibility by setting seeds for all randomness sources."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model_checkpoint(model, optimizer, modelname, dataset_name, batch_size, metrics, logger, prs_enabled):
    """Save trained model with PRS metadata, including warmup_epochs in filename."""
    warmup_epochs = config["warmup_epochs"]
    save_dir = os.path.join("models", "saved", f"{modelname}_{dataset_name}_batch_{batch_size}_warmup_{warmup_epochs}{'_PRS' if prs_enabled else ''}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Model and metadata saved in {save_dir}")

def freeze_final_layer(model, modelname, logger):
    """Freeze the parameters of the final layer in the model."""
    if 'resnet' in modelname.lower():
        final_layer = model.fc
    elif 'vgg' in modelname.lower():
        final_layer = model.classifier[-1]
    elif 'mobilenet' in modelname.lower():
        final_layer = model.classifier
    else:
        try:
            final_layer = model.fc
            logger.info(f"Using model.fc as final layer for {modelname}")
        except AttributeError:
            final_layer = None
            for name, module in reversed(list(model.named_modules())):
                if len(list(module.parameters())) > 0:
                    final_layer = module
                    logger.info(f"Identified final layer as: {name}")
                    break
    
    if final_layer:
        for param in final_layer.parameters():
            param.requires_grad = False
        logger.info(f"Frozen parameters of the final layer for {modelname}")
    else:
        logger.error(f"Could not find final layer to freeze for {modelname}")

def train():
    """Training loop with warm-up, final layer freezing, and PRS regularization."""
    set_seed(config["seed"])

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                logger = setup_logger(modelname, dataset_name, batch_size)
                warmup_epochs = config["warmup_epochs"]
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

                scaler = torch.amp.GradScaler()  # Corrected: Removed "cuda" argument
                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
                activations = {"penultimate": [], "skip_batch": False}

                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                # 🔹 Warm-up stage
                for epoch in tqdm(range(warmup_epochs), desc=f"Warm-up Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
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

                    # 🔹 Prevent test images from influencing PRS calculation
                    hook_handle.remove()
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}%")

                # Check if we have collected activations during warm-up
                if len(activations["penultimate"]) > 0:
                    # 🔹 Compute Major Regions before PRS Regularization
                    major_regions, unique_patterns = compute_major_regions(final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger)
                    save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, modelname, logger, prs_enabled=True, warmup_epochs=config["warmup_epochs"])
                else:
                    logger.error("No activations collected during warm-up. Cannot compute major regions.")
                    continue  # Skip to next configuration

                # 🔹 Freeze final layer and recreate optimizer
                freeze_final_layer(model, modelname, logger)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=0)

                # 🔹 PRS Regularization Stage
                for epoch in tqdm(range(warmup_epochs, config["epochs"]), desc=f"PRS Regularized Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0  # Added epoch_loss tracking

                    activations["penultimate"].clear()
                    batch_labels = []

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        activations["skip_batch"] = False
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                        optimizer.zero_grad()

                        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        if len(activations["penultimate"]) > 0:
                            try:
                                final_activations = torch.cat(activations["penultimate"], dim=0)
                                mrv_loss = compute_mrv_loss(final_activations, labels, major_regions)
                                hamming_loss = compute_hamming_loss(final_activations, labels, major_regions)
                                loss += config["lambda_mrv"] * mrv_loss + config["lambda_hamming"] * hamming_loss
                            except Exception as e:
                                logger.error(f"Error computing regularization loss: {e}")
                                # Continue without regularization if there's an error
                        
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
                    
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")
                
                    # Save model at the final epoch
                    if epoch == config["epochs"] - 1:
                        save_model_checkpoint(model, optimizer, modelname, dataset_name, batch_size, metrics, logger, prs_enabled=True)

    logger.info("Training Complete")

if __name__ == "__main__":
    train()