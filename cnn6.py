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
    register_activation_hook, compute_major_regions,
    save_model_checkpoint, set_seed
)
from utils.logger import setup_logger  
from config import config

def train():
    set_seed(config["seed"])
    results = {}

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):

                logger = setup_logger(modelname, dataset_name, batch_size)
                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size}")

                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, 
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                model = get_model(modelname, input_channels).to(config["device"])
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
                criterion = nn.CrossEntropyLoss()

                metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
                activations = {"penultimate": [], "skip_batch": False}

                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                total_epochs = config["epochs"]
                save_epochs = set(range(50, total_epochs + 1, 50))
                save_epochs.add(total_epochs)

                for epoch in tqdm(range(1, total_epochs + 1), desc=f"Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []
                    skipped_batches = 0

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                        optimizer.zero_grad()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # Always collect activations and labels
                        batch_labels.append(labels.cpu().numpy())

                        if torch.isnan(loss).any():
                            logger.warning(f"NaN in loss at epoch {epoch}, batch {batch_idx}. Skipping.")
                            skipped_batches += 1
                            continue

                        loss.backward()

                        if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                            logger.warning(f"NaN in gradients at epoch {epoch}, batch {batch_idx}. Skipping update.")
                            skipped_batches += 1
                            continue

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()

                        epoch_loss += loss.item()
                        _, predicted = outputs.max(1)
                        correct_train += (predicted == labels).sum().item()
                        total_train += labels.size(0)

                    if activations["penultimate"]:
                        all_activations = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                        all_labels = np.concatenate(batch_labels, axis=0)
                        prs_ratio = compute_unique_activations(all_activations, logger) / all_labels.shape[0]
                    else:
                        logger.warning("No activations collected this epoch — skipping PRS computation.")
                        prs_ratio = 0

                    train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
                    test_accuracy = evaluate(model, test_loader, config["device"])

                    metrics["epoch"].append(epoch)
                    metrics["train_accuracy"].append(train_accuracy)
                    metrics["test_accuracy"].append(test_accuracy)
                    metrics["prs_ratios"].append(prs_ratio)

                    logger.info(f"Epoch {epoch}/{total_epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | PRS: {prs_ratio:.4f} | Skipped Batches: {skipped_batches}")

                    if epoch in save_epochs and activations["penultimate"]:
                        major_regions, unique_patterns = compute_major_regions(all_activations, all_labels, num_classes=10, logger=logger)
                        save_model_checkpoint(
                            model, optimizer, modelname, dataset_name, batch_size,
                            metrics, logger, config=config,
                            extra_tag=None, epoch=epoch,
                            major_regions=major_regions, unique_patterns=unique_patterns
                        )

                hook_handle.remove()
                results[f"{dataset_name}_batch_{batch_size}"] = metrics

    logger.info("Training Complete")

if __name__ == "__main__":
    train()
