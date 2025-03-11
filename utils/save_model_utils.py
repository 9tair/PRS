import os
import torch
import json

def save_model_checkpoint(
    model, optimizer, modelname, dataset_name, batch_size, 
    metrics, logger, scheduler=None, prs_enabled=False, 
    config=None, extra_tag=None, epoch=None, major_regions=None, unique_patterns=None
):
    """Generalized function to save model, optimizer, scheduler, and metadata.

    Args:
        model (torch.nn.Module): The trained model to save.
        optimizer (torch.optim.Optimizer): Optimizer state.
        modelname (str): Name of the model.
        dataset_name (str): Name of the dataset.
        batch_size (int): Batch size used in training.
        metrics (dict): Dictionary containing evaluation metrics.
        logger (logging.Logger): Logger for logging messages.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler state. Defaults to None.
        prs_enabled (bool, optional): Whether PRS is enabled. Defaults to False.
        config (dict, optional): Configuration dictionary, required for warmup_epochs. Defaults to None.
        extra_tag (str, optional): Additional tag to append to the directory name. Defaults to None.
        epoch (int, optional): The current epoch number for creating subfolders.
        major_regions (dict, optional): Major region data to save.
        unique_patterns (dict, optional): Unique pattern data to save.

    """
    # Construct base save directory name
    save_dir = f"{modelname}_{dataset_name}_batch_{batch_size}"

    if config and "warmup_epochs" in config:
        save_dir += f"_warmup_{config['warmup_epochs']}"

    if prs_enabled:
        save_dir += "_PRS"

    if extra_tag:
        save_dir += f"_{extra_tag}"

    # Define base save path
    base_save_path = os.path.join("models", "saved", save_dir)
    os.makedirs(base_save_path, exist_ok=True)

    # Save config & metrics in the base folder (only once)
    if config:
        config_path = os.path.join(base_save_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    if metrics:
        metrics_path = os.path.join(base_save_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # If epoch is specified, save in subfolder
    if epoch is not None:
        epoch_dir = os.path.join(base_save_path, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pth"))
        torch.save(model.state_dict(), os.path.join(epoch_dir, "weights.pth"))  # Saving model weights separately
        torch.save(optimizer.state_dict(), os.path.join(epoch_dir, "optimizer.pth"))

        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(epoch_dir, "scheduler.pth"))

        # Save major regions and unique patterns
        if major_regions is not None:
            with open(os.path.join(epoch_dir, "major_regions.json"), "w") as f:
                json.dump(major_regions, f, indent=4)

        if unique_patterns is not None:
            with open(os.path.join(epoch_dir, "unique_patterns.json"), "w") as f:
                json.dump(unique_patterns, f, indent=4)

        logger.info(f"Model and metadata saved in {epoch_dir}")
