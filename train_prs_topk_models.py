import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy # For deepcopying model states and other objects

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
    
    # Add this to config or define it here
    TOP_K_PRS_SAVE = config.get("top_k_prs_save", 5) # Number of top models to save by PRS

    for modelname in tqdm(config['models'], desc="Model Loop"):
        for dataset_name in tqdm(config["datasets"], desc="Dataset Loop"):
            for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):

                logger = setup_logger(modelname, dataset_name, batch_size)
                logger.info(f"Starting Training | Model: {modelname} | Dataset: {dataset_name} | Batch Size: {batch_size}")

                train_dataset, test_dataset, input_channels = get_datasets(dataset_name)
                
                # Determine num_classes (important for compute_major_regions)
                if hasattr(train_dataset, 'classes'):
                    num_classes = len(train_dataset.classes)
                elif dataset_name.lower() in ["cifar10", "mnist", "fashionmnist"]: # common datasets
                    num_classes = 10
                elif dataset_name.lower() == "cifar100":
                    num_classes = 100
                else:
                    # Fallback, or raise an error if num_classes is critical and unknown
                    logger.warning(f"Could not determine num_classes for {dataset_name}. Defaulting to 10 for major regions.")
                    num_classes = 10


                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True,
                    worker_init_fn=lambda _: np.random.seed(config["seed"]),
                    generator=torch.Generator().manual_seed(config["seed"])
                )
                test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

                model = get_model(modelname, input_channels).to(config["device"])
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
                criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))

                metrics_history = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
                activations = {"penultimate": [], "skip_batch": False}

                hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)

                total_epochs = config["epochs"]
                
                # List to store (prs_ratio, epoch, model_state_dict, optimizer_state_dict, epoch_metrics, all_activations_epoch, all_labels_epoch)
                # We sort by prs_ratio (ascending)
                top_prs_models_data = []

                for epoch in tqdm(range(1, total_epochs + 1), desc=f"Training {dataset_name} | Batch {batch_size}"):
                    model.train()
                    epoch_loss, correct_train, total_train = 0, 0, 0
                    activations["penultimate"].clear()
                    batch_labels_for_epoch = [] # Changed variable name for clarity
                    skipped_batches = 0
                    
                    all_activations_this_epoch = None # To store activations for this specific epoch
                    all_labels_this_epoch = None    # To store labels for this specific epoch

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                        optimizer.zero_grad()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        batch_labels_for_epoch.append(labels.cpu().numpy())

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

                    prs_ratio = 0 # Default PRS
                    if activations["penultimate"]:
                        all_activations_this_epoch = torch.cat(activations["penultimate"], dim=0).cpu().numpy()
                        all_labels_this_epoch = np.concatenate(batch_labels_for_epoch, axis=0)
                        prs_ratio = compute_unique_activations(all_activations_this_epoch, logger) / all_labels_this_epoch.shape[0]
                    else:
                        logger.warning("No activations collected this epoch — skipping PRS computation.")
                        # all_activations_this_epoch and all_labels_this_epoch will remain None

                    train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
                    test_accuracy = evaluate(model, test_loader, config["device"])

                    metrics_history["epoch"].append(epoch)
                    metrics_history["train_accuracy"].append(train_accuracy)
                    metrics_history["test_accuracy"].append(test_accuracy)
                    metrics_history["prs_ratios"].append(prs_ratio)
                    
                    current_epoch_metrics = {
                        "epoch": epoch,
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                        "prs_ratio": prs_ratio,
                        "loss": epoch_loss # Adding loss as well, might be useful
                    }

                    logger.info(f"Epoch {epoch}/{total_epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | PRS: {prs_ratio:.4f} | Skipped Batches: {skipped_batches}")

                    # Manage top K PRS models
                    # Only consider if PRS was computable (i.e., activations were collected)
                    if (
                        all_activations_this_epoch is not None and 
                        all_labels_this_epoch is not None and
                        train_accuracy >= 70.0 and 
                        test_accuracy >= 70.0
                    ):
                        # Store necessary data, deepcopy states to prevent modification by subsequent training
                        model_state = copy.deepcopy(model.state_dict())
                        optimizer_state = copy.deepcopy(optimizer.state_dict())
                        
                        # Create an entry for this epoch
                        entry = (
                            prs_ratio, 
                            epoch, 
                            model_state, 
                            optimizer_state,
                            copy.deepcopy(current_epoch_metrics), # Save snapshot of current epoch's metrics
                            copy.deepcopy(all_activations_this_epoch), # Deepcopy numpy arrays
                            copy.deepcopy(all_labels_this_epoch)
                        )

                        if len(top_prs_models_data) < TOP_K_PRS_SAVE:
                            top_prs_models_data.append(entry)
                            top_prs_models_data.sort(key=lambda x: x[0]) # Sort by PRS (ascending)
                        elif prs_ratio < top_prs_models_data[-1][0]: # If current PRS is better than the worst in top_k
                            top_prs_models_data.pop() # Remove the worst (largest PRS)
                            top_prs_models_data.append(entry)
                            top_prs_models_data.sort(key=lambda x: x[0]) # Re-sort

                hook_handle.remove()
                
                # After all epochs for this config, save the top K models
                logger.info(f"Saving top {len(top_prs_models_data)} models based on lowest PRS ratio...")
                for rank, data_tuple in enumerate(top_prs_models_data):
                    (
                        saved_prs,
                        saved_epoch,
                        saved_model_state,
                        saved_optimizer_state,
                        saved_epoch_metrics,
                        saved_activations,
                        saved_labels,
                    ) = data_tuple

                    # Temporarily load the saved state into the model and optimizer
                    # This is necessary because compute_major_regions might interact with the model
                    # or save_model_checkpoint might save the current model object
                    model.load_state_dict(saved_model_state)
                    optimizer.load_state_dict(saved_optimizer_state)
                    
                    # Compute major regions using the activations and labels from that specific epoch
                    # Ensure num_classes is correctly determined or passed
                    major_regions, unique_patterns = compute_major_regions(
                        saved_activations, saved_labels, num_classes=num_classes, logger=logger
                    )
                    
                    logger.info(f"Saving model from epoch {saved_epoch} (Rank {rank+1} PRS: {saved_prs:.4f})")
                    save_model_checkpoint(
                        model,  # Model now has the loaded state
                        optimizer, # Optimizer now has the loaded state
                        modelname,
                        dataset_name,
                        batch_size,
                        saved_epoch_metrics, # Pass metrics for this specific epoch
                        logger,
                        config=config,
                        extra_tag=f"norm_ls_top_prs_rank{rank+1}", # Tag to identify its rank
                        epoch=saved_epoch, # The actual epoch number of this saved model
                        major_regions=major_regions,
                        unique_patterns=unique_patterns
                    )

                results[f"{modelname}_{dataset_name}_batch_{batch_size}"] = metrics_history # Save full history for this run

    logger.info("Training Complete")
    # Potentially save the 'results' dictionary here if needed for overall analysis
    # e.g., import pickle; with open("all_training_results.pkl", "wb") as f: pickle.dump(results, f)

if __name__ == "__main__":
    # Example config (ensure your actual config.py has these or similar)
    # This is just for making the script runnable if config.py is missing these specific keys
    if "top_k_prs_save" not in config:
        config["top_k_prs_save"] = 10 # Default if not in config
    if "seed" not in config:
        config["seed"] = 42
    if "test_batch_size" not in config:
        config["test_batch_size"] = 1000
    if "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    train()