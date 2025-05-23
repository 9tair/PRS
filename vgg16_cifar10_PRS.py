import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

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


def load_checkpoint_if_exists(model, optimizer, modelname, dataset_name, batch_size, logger):
    """Loads a model checkpoint from `epoch_{warmup_epochs}` instead of the latest epoch."""
    checkpoint_path = os.path.join(
        "models", "saved", f"{modelname}_{dataset_name}_batch_{batch_size}"
    )
    warmup_epoch = config["warmup_epochs"]
    epoch_path = os.path.join(checkpoint_path, f"epoch_{warmup_epoch}")

    if os.path.exists(epoch_path):
        logger.info(f"Loading checkpoint from epoch {warmup_epoch}: {epoch_path}")
        try:
            map_loc = config["device"]
            model.load_state_dict(torch.load(os.path.join(epoch_path, "model.pth"), map_location=map_loc))
            optimizer.load_state_dict(torch.load(os.path.join(epoch_path, "optimizer.pth"), map_location=map_loc))
            scheduler_state = (
                torch.load(os.path.join(epoch_path, "scheduler.pth"), map_location=map_loc)
                if os.path.exists(os.path.join(epoch_path, "scheduler.pth")) else None
            )
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
    """Enhanced training loop with warm-up, PRS regularization, and stability features."""
    set_seed(config["seed"])
    results = {}

    # Stability-focused hyperparams
    base_lr = 5e-4
    weight_decay = 1e-4

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
                # ─── Setup ───────────────────────────────────────────────
                logger = setup_logger(modelname, dataset_name, batch_size)
                warmup_epochs = config["warmup_epochs"]
                total_epochs = config["epochs"]
                save_epochs = set(range(warmup_epochs, total_epochs + 1, 10))
                save_epochs.add(total_epochs)

                logger.info(
                    f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | "
                    f"Batch Size: {batch_size} | Warmup: {warmup_epochs}"
                )
                logger.info(f"Using lr={base_lr}, weight_decay={weight_decay}")

                # ─── Data ────────────────────────────────────────────────
                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True,
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                # ─── Model / Optimizer / AMP Setup ──────────────────────
                device  = torch.device(config["device"])
                model   = get_model(modelname, input_channels).to(device)
                model   = initialize_weights(model)

                optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
                scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=base_lr / 10)
                criterion  = nn.CrossEntropyLoss()

                use_amp    = torch.cuda.is_available() and config.get("use_amp", True)
                scaler     = torch.amp.GradScaler(enabled=use_amp)

                # ─── Metrics & Hook ─────────────────────────────────────
                metrics = {
                    "epoch": [], "train_accuracy": [], "test_accuracy": [],
                    "prs_ratios": [], "learning_rates": [], "mrv_loss": [], "hamming_loss": []
                }
                activations = {"penultimate": [], "skip_batch": False, "current": None}
                hook_handle = register_activation_hook(
                    model, activations, modelname, dataset_name, batch_size, logger
                )

                # ─── Possibly Load Checkpoint ───────────────────────────
                ckpt_exists, model, optimizer, sched_state, major_regions, unique_patterns = \
                    load_checkpoint_if_exists(model, optimizer, modelname, dataset_name, batch_size, logger)

                if sched_state is not None:
                    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=base_lr/10)
                    try:
                        scheduler.load_state_dict(sched_state)
                        logger.info("Loaded scheduler state from checkpoint")
                    except Exception as e:
                        logger.warning(f"Could not load scheduler state: {e}")

                start_epoch = warmup_epochs if ckpt_exists else 0

                # ─── Warm-up Stage ───────────────────────────────────────
                if not ckpt_exists:
                    logger.info("Starting warm-up training.")
                    for epoch in tqdm(range(warmup_epochs), desc="Warm-up"):
                        model.train()
                        epoch_loss, correct, total = 0.0, 0, 0
                        batch_labels = []

                        lr = scheduler.get_last_lr()[0]
                        metrics["learning_rates"].append(lr)
                        logger.info(f"Warm-up Epoch {epoch+1}/{warmup_epochs} | LR {lr:.6f}")

                        for batch_idx, (inputs, labels) in enumerate(train_loader):
                            inputs, labels = inputs.to(device), labels.to(device)
                            optimizer.zero_grad()

                            dtype = "cuda" if use_amp else "cpu"
                            with torch.amp.autocast(device_type=dtype, enabled=use_amp):
                                outputs = model(inputs)
                                loss    = criterion(outputs, labels)

                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()

                            epoch_loss += loss.item()
                            preds = outputs.argmax(dim=1)
                            correct   += (preds == labels).sum().item()
                            total     += labels.size(0)
                            batch_labels.append(labels.cpu().numpy())

                        scheduler.step()
                        train_acc = 100 * correct / total if total else 0

                        # collect PRS ratio
                        if activations["penultimate"]:
                            acts_np = torch.cat(activations["penultimate"]).cpu().numpy()
                            prs_ratio = compute_unique_activations(acts_np, logger) / len(train_dataset)
                        else:
                            prs_ratio = 0.0
                            logger.warning("No activations collected in warm-up.")

                        metrics["epoch"].append(epoch+1)
                        metrics["train_accuracy"].append(train_acc)
                        metrics["prs_ratios"].append(prs_ratio)

                        hook_handle.remove()
                        test_acc = evaluate(model, test_loader, device)
                        hook_handle = register_activation_hook(
                            model, activations, modelname, dataset_name, batch_size, logger
                        )
                        metrics["test_accuracy"].append(test_acc)

                        logger.info(
                            f"Warm-up {epoch+1}/{warmup_epochs} | Loss {epoch_loss:.4f} | "
                            f"TrAcc {train_acc:.2f}% | TeAcc {test_acc:.2f}% | PRS {prs_ratio:.4f}"
                        )

                    # compute initial major regions
                    if activations["penultimate"]:
                        acts_np = torch.cat(activations["penultimate"]).cpu().numpy()
                        lbls_np = np.concatenate(batch_labels, axis=0)
                        major_regions, unique_patterns = compute_major_regions(
                            acts_np, lbls_np, num_classes=10, logger=logger
                        )
                    else:
                        logger.error("Warm-up failed to collect activations; aborting.")
                        return

                # ─── PRS Stage ───────────────────────────────────────────
                logger.info("Freezing final layer & starting PRS phase.")
                freeze_final_layer(model, modelname, logger)
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=base_lr, weight_decay=weight_decay
                )
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=(total_epochs - warmup_epochs), eta_min=base_lr/10
                )

                for epoch in tqdm(range(start_epoch, total_epochs), desc="PRS"):
                    model.train()
                    epoch_loss, correct, total = 0.0, 0, 0
                    total_mrv, total_ham = 0.0, 0.0
                    activations["penultimate"].clear()
                    batch_labels = []

                    lr = scheduler.get_last_lr()[0]
                    metrics["learning_rates"].append(lr)
                    logger.info(f"PRS Epoch {epoch+1}/{total_epochs} | LR {lr:.6f}")

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()

                        dtype = "cuda" if use_amp else "cpu"
                        with torch.amp.autocast(device_type=dtype, enabled=use_amp):
                            outputs = model(inputs)
                            loss    = criterion(outputs, labels)

                        if activations["current"] is not None:
                            mrv = compute_mrv_loss(activations["current"], labels, major_regions, logger)
                            ham = compute_hamming_loss(activations["current"], labels, major_regions, logger)
                            loss = loss + config["lambda_mrv"]*mrv + config["lambda_hamming"]*ham
                            total_mrv += float(mrv.item())
                            total_ham += float(ham.item())

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        epoch_loss += loss.item()
                        preds = outputs.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total   += labels.size(0)
                        batch_labels.append(labels.cpu().numpy())

                        if activations["current"] is not None:
                            activations["penultimate"].append(activations["current"].detach().cpu())

                    scheduler.step()
                    train_acc = 100 * correct / total if total else 0
                    prs_ratio = (
                        compute_unique_activations(
                            torch.cat(activations["penultimate"]).cpu().numpy(), logger
                        ) / len(train_dataset)
                        if activations["penultimate"] else 0.0
                    )

                    hook_handle.remove()
                    test_acc = evaluate(model, test_loader, device)
                    hook_handle = register_activation_hook(
                        model, activations, modelname, dataset_name, batch_size, logger
                    )

                    metrics["epoch"].append(epoch+1)
                    metrics["train_accuracy"].append(train_acc)
                    metrics["test_accuracy"].append(test_acc)
                    metrics["prs_ratios"].append(prs_ratio)
                    metrics["mrv_loss"].append(total_mrv / len(train_loader))
                    metrics["hamming_loss"].append(total_ham / len(train_loader))

                    logger.info(
                        f"PRS {epoch+1}/{total_epochs} | Loss {epoch_loss:.4f} | "
                        f"TrAcc {train_acc:.2f}% | TeAcc {test_acc:.2f}% | "
                        f"MRV {total_mrv/len(train_loader):.6f} | HAM {total_ham/len(train_loader):.6f}"
                    )

                    if (epoch+1) in save_epochs:
                        # recompute and save regions/checkpoint
                        acts_np = torch.cat(activations["penultimate"]).cpu().numpy()
                        lbls_np = np.concatenate(batch_labels, axis=0)
                        major_regions, unique_patterns = compute_major_regions(
                            acts_np, lbls_np, num_classes=10, logger=logger
                        )
                        save_model_checkpoint(
                            model, optimizer, modelname, dataset_name, batch_size,
                            metrics, logger, config=config,
                            extra_tag=None, epoch=epoch, prs_enabled=True,
                            major_regions=major_regions, unique_patterns=unique_patterns
                        )

                hook_handle.remove()
                # save final metrics
                os.makedirs(config["results_save_path"], exist_ok=True)
                results[f"{dataset_name}_batch_{batch_size}"] = metrics

    logger.info("Training Complete")


if __name__ == "__main__":
    train()
