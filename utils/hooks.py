from utils.logger import setup_logger  # Import dynamic logger
import torch.nn as nn
import torch

def register_activation_hook(model, activations, model_name, dataset_name, batch_size, logger):
    """
    Register a hook on the penultimate layer to detect NaNs and log them.

    Args:
        model (torch.nn.Module): Model to register the hook on.
        activations (dict): Dictionary storing activations and a skip flag.
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        batch_size (int): Training batch size.

    Returns:
        hook_handle: Hook reference for later removal.
    """

    def hook_fn(module, input, output):
        """Hook function to track activations and detect NaNs."""
        if torch.isnan(output).any():  # Detect NaNs
            msg = f"NaN detected | Model: {model_name} | Dataset: {dataset_name} | Batch Size: {batch_size}"
            logger.warning(msg)
            activations["skip_batch"] = True  # Flag to skip this batch

        activations["penultimate"].append(output.detach().cpu().numpy())

    activations["skip_batch"] = False  # Initialize flag

    # Identify and register hook on the correct layer
    hook_handle = None

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):  
        penultimate_layer_index = len(model.classifier) - 2
        hook_handle = model.classifier[penultimate_layer_index].register_forward_hook(hook_fn)
        logger.info(f"Hook registered on classifier layer {penultimate_layer_index}.")

    elif hasattr(model, "fc"):  
        if isinstance(model.fc, nn.Sequential):
            penultimate_layer_index = len(model.fc) - 2
            hook_handle = model.fc[penultimate_layer_index].register_forward_hook(hook_fn)
        else:
            hook_handle = model.fc.register_forward_hook(hook_fn)
        logger.info(f"Hook registered on fully connected (fc) layer.")

    else:
        logger.error("Hook registration failed: Model structure unknown.")
        raise ValueError("Hook could not be registered. Model structure unknown.")

    return hook_handle
