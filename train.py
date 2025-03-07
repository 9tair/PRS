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

def set_seed(seed):
    """Ensure reproducibility by setting seeds for all randomness sources."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def train():
    """Main training loop handling different datasets and batch sizes."""
    set_seed(config["seed"])
    results = {}

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                # Initialize dynamic logger for this setting
                logger = setup_logger(modelname, dataset_name, batch_size)
                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size}")

                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                train_dataset_size = len(train_dataset)  # Avoid repeated calls

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, 
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                model = get_model(modelname, input_channels).to(config["device"])
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
                criterion = nn.CrossEntropyLoss()

                scaler = torch.amp.GradScaler("cuda")

                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
                activations = {"penultimate": [], "skip_batch": False}

                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

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
                            loss = criterion(outputs, labels)

                        if activations["skip_batch"]:
                            logger.warning(f"Skipping NaN batch | Epoch: {epoch+1} | Batch: {batch_idx}")
                            continue  # Skip backprop entirely

                        scaler.scale(loss).backward()

                        if any(torch.isnan(param.grad).any() for param in model.parameters()):
                            logger.warning(f"NaN detected in gradients. Skipping update | Epoch: {epoch+1} | Batch: {batch_idx}")
                            continue  # Skip optimizer step

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                        scaler.step(optimizer)
                        scaler.update()
                        
                        epoch_loss += loss.item()
                        batch_labels.append(labels.cpu().numpy())

                        _, predicted = outputs.max(1)
                        correct_train += (predicted == labels).sum().item()
                        total_train += labels.size(0)
                    
                    # 🔹 Keep activations as tensors longer, convert only when necessary
                    final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                    final_epoch_labels = np.concatenate(batch_labels, axis=0)

                    # Compute PRS Ratio
                    prs_ratio = compute_unique_activations(final_epoch_activations, logger) / train_dataset_size
                    metrics["prs_ratios"].append(prs_ratio)

                    # Evaluate on Test Set
                    test_accuracy = evaluate(model, test_loader, config["device"])
                    train_accuracy = 100 * correct_train / total_train
                    
                    # Store Metrics
                    metrics["epoch"].append(epoch + 1)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    
                    logger.info(f"Epoch {epoch+1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}%")

                hook_handle.remove()  # 🔹 Ensure hook is removed after training

                # 🔹 Compute and Save MR/ER using only LAST epoch activations
                major_regions, unique_patterns = compute_major_regions(final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger)
                
                save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, modelname, logger, warmup_epochs=config["warmup_epochs"])

                # 🔹 Ensure results directory exists
                results_save_path = config["results_save_path"]
                os.makedirs(results_save_path, exist_ok=True)

                results[f"{dataset_name}_batch_{batch_size}"] = metrics

                save_model_checkpoint(model, optimizer, modelname, dataset_name, batch_size, metrics, logger)

    logger.info("Training Complete")

if __name__ == "__main__":
    train()
