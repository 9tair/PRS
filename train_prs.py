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
from utils.regularization import compute_mrv_loss, compute_hamming_loss
from config import config

def set_seed(seed):
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

def train():
    """Training loop with warm-up and PRS regularization."""
    set_seed(config["seed"])
    results = {}

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                
                # Initialize logger
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

                scaler = torch.amp.GradScaler("cuda")

                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
                epoch_activations, epoch_labels = [], []

                activations = {"penultimate": [], "skip_batch": False}
                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                # 🔹 Warm-up stage
                for epoch in tqdm(range(warmup_epochs), desc=f"Warm-up Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []

                    logger.info(f"Warm-up Epoch {epoch+1}/{warmup_epochs}")

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        activations["skip_batch"] = False

                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                        optimizer.zero_grad()
                        
                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        scaler.scale(loss).backward()
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
                    prs_ratio = compute_unique_activations(final_epoch_activations, logger) / len(train_dataset)
                    metrics["prs_ratios"].append(prs_ratio)

                # 🔹 Compute Major Regions before PRS Regularization
                major_regions, unique_patterns = compute_major_regions(final_epoch_activations, final_epoch_labels, num_classes=10, logger=logger)
                save_major_regions(major_regions, unique_patterns, dataset_name, batch_size, modelname, logger, prs_enabled=False, warmup_epochs=config["warmup_epochs"])

                # 🔹 PRS Regularization Stage
                for epoch in tqdm(range(warmup_epochs, config["epochs"]), desc=f"PRS Regularized Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        activations["skip_batch"] = False

                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                        batch_labels.append(labels.cpu().numpy())
                        optimizer.zero_grad()

                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            cross_entropy_loss = criterion(outputs, labels)

                        final_activations = torch.cat(activations["penultimate"], dim=0)

                        # Compute PRS Loss
                        mrv_loss = compute_mrv_loss(final_activations, labels, major_regions)
                        hamming_loss = compute_hamming_loss(final_activations, labels, major_regions)
                        total_loss = config["lambda_std"] * cross_entropy_loss + config["lambda_mrv"] * mrv_loss + config["lambda_hamming"] * hamming_loss

                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        epoch_loss += total_loss.item()

                    # Compute PRS Ratio
                    final_epoch_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                    final_epoch_labels = np.concatenate(batch_labels, axis=0)
                    prs_ratio = compute_unique_activations(final_epoch_activations, logger) / len(train_dataset)
                    metrics["prs_ratios"].append(prs_ratio)

                # Save results
                results_save_path = config["results_save_path"]
                os.makedirs(results_save_path, exist_ok=True)

                results[f"{dataset_name}_batch_{batch_size}_warmup_{warmup_epochs}"] = metrics
                with open(os.path.join(results_save_path, f"PRS_metrics_{modelname}_{dataset_name}_batch_{batch_size}_warmup_{warmup_epochs}.json"), "w") as f:
                    json.dump(metrics, f, indent=4)

                save_model_checkpoint(model, optimizer, modelname, dataset_name, batch_size, metrics, logger, prs_enabled=True)

    logger.info("PRS Regularized Training Complete")

if __name__ == "__main__":
    train()