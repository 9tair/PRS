from utils.logger import global_logger  # Import from logger.py to avoid circular import
logger = global_logger  # Use the global logger

import torch.nn as nn
import torch

def register_activation_hook(model, activations, model_name, dataset_name, batch_size):
    """Register a hook on the penultimate layer to detect NaNs and log them."""

    def hook_fn(module, input, output):
        if torch.isnan(output).any():  # Detect NaNs
            msg = f"⚠️ NaN detected | Model: {model_name} | Dataset: {dataset_name} | Batch Size: {batch_size}"
            logger.warning(msg)  # This will use logging from train.py
            activations["skip_batch"] = True  # Flag to skip NaN batches
        
        activations["penultimate"].append(output.detach().cpu().numpy())

    activations["skip_batch"] = False  # Initialize flag

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):  
        penultimate_layer_index = len(model.classifier) - 2
        hook_handle = model.classifier[penultimate_layer_index].register_forward_hook(hook_fn)

    elif hasattr(model, "fc"):  
        if isinstance(model.fc, nn.Sequential):
            penultimate_layer_index = len(model.fc) - 2
            hook_handle = model.fc[penultimate_layer_index].register_forward_hook(hook_fn)
        else:
            hook_handle = model.fc.register_forward_hook(hook_fn)

    else:
        raise ValueError("Hook could not be registered. Model structure unknown.")

    return hook_handle
