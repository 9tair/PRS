import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from utils import (
    get_datasets, evaluate, compute_unique_activations, 
    register_activation_hook, compute_major_regions,
    save_model_checkpoint, set_seed
)
from utils.logger import setup_logger
from config import config


def pgd_attack(model, images, labels, eps=0.031, alpha=0.007, iters=10):
    ori_images = images.detach()
    adv_images = ori_images.clone().detach()

    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = adv_images.grad.data
        adv_images = adv_images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    return adv_images


def evaluate_robust(model, test_loader, eps=0.031, alpha=0.007, iters=20):
    correct, total = 0, 0
    model.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
        # Enable gradient tracking for PGD
        adv_inputs = pgd_attack(model, inputs, labels, eps, alpha, iters)
        with torch.no_grad():  # Only disable grads for forward pass
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_adversarial():
    set_seed(config["seed"])

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):

                logger = setup_logger(f"{modelname}_advtrain", dataset_name, batch_size)
                logger.info(f"[Adversarial Training] Model: {modelname} | Dataset: {dataset_name} | Batch: {batch_size}")

                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                model = get_model(modelname, input_channels).to(config["device"])
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
                criterion = nn.CrossEntropyLoss()

                activations = {"penultimate": [], "skip_batch": False}
                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                metrics = {
                    "epoch": [],
                    "train_accuracy": [],
                    "test_accuracy": [],
                    "robust_accuracy": [],
                    "prs_ratio": []
                }

                total_epochs = config["epochs"]
                warmup_epochs = config.get("warmup_epochs", 50)
                logger.info(f"Warm-up phase: {warmup_epochs} epochs")

                for epoch in tqdm(range(1, total_epochs + 1), desc="Training Epochs"):
                    model.train()
                    epoch_loss, correct, total = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels = []

                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                        if epoch <= warmup_epochs:
                            adv_inputs = inputs
                        else:
                            adv_inputs = pgd_attack(model, inputs, labels)

                        optimizer.zero_grad()
                        outputs = model(adv_inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()

                        epoch_loss += loss.item()
                        _, preds = outputs.max(1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                        batch_labels.append(labels.cpu().numpy())

                    train_acc = 100 * correct / total
                    test_acc = evaluate(model, test_loader, config["device"])
                    robust_acc = evaluate_robust(model, test_loader) if epoch > warmup_epochs else 0

                    if activations["penultimate"]:
                        all_acts = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                        n_samples = all_acts.shape[0]  # trust activation count
                        try:
                            n_unique = compute_unique_activations(all_acts, logger)
                            prs_ratio = n_unique / n_samples

                            logger.debug(f"[DEBUG] Epoch {epoch}: #Activations collected: {len(activations['penultimate'])}")
                            logger.debug(f"[DEBUG] Activations shape: {all_acts.shape}")
                            logger.debug(f"[DEBUG] Unique patterns: {n_unique}, Samples: {n_samples}, PRS: {prs_ratio:.4f}")
                            logger.debug(f"[DEBUG] Activation stats: min={all_acts.min():.4f}, max={all_acts.max():.4f}, mean={all_acts.mean():.4f}")
                        except Exception as e:
                            logger.warning(f"[PRS ERROR] Failed to compute PRS: {e}")
                            prs_ratio = 0
                    else:
                        logger.warning("No activations collected — skipping PRS computation.")
                        prs_ratio = 0

                    logger.info(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                                f"Test Acc: {test_acc:.2f}% | Robust Acc: {robust_acc:.2f}% | PRS: {prs_ratio:.4f}")

                    metrics["epoch"].append(epoch)
                    metrics["train_accuracy"].append(train_acc)
                    metrics["test_accuracy"].append(test_acc)
                    metrics["robust_accuracy"].append(robust_acc)
                    metrics["prs_ratio"].append(prs_ratio)

                hook_handle.remove()

                # Save JSON
                save_path = f"results/{modelname}_{dataset_name}_batch_{batch_size}_advtrain_metrics.json"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(metrics, f, indent=4)

                logger.info(f"Saved metrics to {save_path}")


if __name__ == "__main__":
    train_adversarial()
