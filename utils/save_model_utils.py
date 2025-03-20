import os
import torch
import json
import numpy as np

def convert_float16_to_float32(data):
    """Recursively convert float16 values in a dictionary to float32."""
    if isinstance(data, dict):
        return {k: convert_float16_to_float32(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_float16_to_float32(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(convert_float16_to_float32(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.float().tolist()  # Convert Tensor to list of float32
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32).tolist()  # Convert NumPy array to list of float32
    elif isinstance(data, np.number):
        return float(data)  # Convert NumPy scalar types to Python float
    elif hasattr(data, 'dtype') and str(data.dtype) == 'float16':
        return float(data)  # Handle any other float16 types
    return data

def save_model_checkpoint(
    model, optimizer, modelname, dataset_name, batch_size, 
    metrics, logger, scheduler=None, prs_enabled=False, 
    config=None, extra_tag=None, epoch=None, major_regions=None, unique_patterns=None
):
    """Generalized function to save model, optimizer, scheduler, and metadata."""
    
    # Construct base save directory name
    save_dir = f"{modelname}_{dataset_name}_batch_{batch_size}"
    if prs_enabled:
        save_dir += f"_warmup_{config['warmup_epochs']}_PRS"
    if extra_tag:
        save_dir += f"_{extra_tag}"

    # Define base save path
    base_save_path = os.path.join("models", "saved", save_dir)
    os.makedirs(base_save_path, exist_ok=True)

    # Save config & metrics in the base folder (only once)
    config_path = os.path.join(base_save_path, "config.json")
    metrics_path = os.path.join(base_save_path, "metrics.json")

    if config and not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(convert_float16_to_float32(config), f, indent=4)  # Convert before saving

    if metrics:
        with open(metrics_path, "w") as f:
            json.dump(convert_float16_to_float32(metrics), f, indent=4)  # Convert before saving

    # If epoch is specified, save in subfolder
    if epoch is not None:
        epoch_dir = os.path.join(base_save_path, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Save model weights
        torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pth"))
        torch.save(model.state_dict(), os.path.join(epoch_dir, "weights.pth"))  
        
        # Save optimizer and scheduler states
        torch.save(optimizer.state_dict(), os.path.join(epoch_dir, "optimizer.pth"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(epoch_dir, "scheduler.pth"))

        # Save major regions and unique patterns
        if major_regions is not None:
            with open(os.path.join(epoch_dir, "major_regions.json"), "w") as f:
                json.dump(convert_float16_to_float32(major_regions), f, indent=4)  

        if unique_patterns is not None:
            with open(os.path.join(epoch_dir, "unique_patterns.json"), "w") as f:
                json.dump(convert_float16_to_float32(unique_patterns), f, indent=4)  

        logger.info(f"Model and metadata saved in {epoch_dir}")