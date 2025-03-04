import torch.nn as nn
import torch

def register_activation_hook(model, activations, model_name, dataset_name, batch_size, logger):
    """
    Registers a forward hook on the correct penultimate layer for activation tracking.

    Args:
        model (torch.nn.Module): The model to register the hook on.
        activations (dict): Dictionary storing activations and a skip flag.
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        batch_size (int): Training batch size.
        logger: Logger instance.

    Returns:
        hook_handle: Hook reference for later removal.
    """

    def hook_fn(module, input, output):
        """Hook function to track activations and detect NaNs."""
        if torch.isnan(output).any():
            logger.warning(f"NaN detected | Model: {model_name} | Dataset: {dataset_name} | Batch Size: {batch_size}")
            activations["skip_batch"] = True  # 🚨 Skip batch if NaNs detected

        activations["penultimate"].append(output.detach())  # 🔹 Keep activations in torch format

    activations["skip_batch"] = False  
    hook_handle = None

    if model_name == "VGG16":
        # 🔹 Correctly register the hook on the **penultimate fully connected layer**
        layer_index = 3  # classifier[5] = ReLU before final FC layer
        hook_handle = model.classifier[layer_index].register_forward_hook(hook_fn)
        logger.info(f"Hook registered on VGG16 classifier[{layer_index}] (penultimate FC).")

    elif model_name == "ResNet18":
        # Previously, hook was placed on model.fc (incorrect)
        # Now, correctly placing the hook on the last **residual block before global average pooling**
        hook_handle = model.layer4[-1].register_forward_hook(hook_fn)
        logger.info("Hook registered on ResNet18 layer4[-1] (penultimate layer before GAP).")

    elif model_name == "CNN-6":
        # Register the hook on the second-to-last layer of the classifier
        hook_handle = model.classifier[-2].register_forward_hook(hook_fn)
        logger.info("Hook registered on CNN-6 penultimate FC.")

    else:
        logger.error(f"Hook registration failed: Unknown model structure {model_name}.")
        raise ValueError("Hook could not be registered. Model structure unknown.")

    return hook_handle
