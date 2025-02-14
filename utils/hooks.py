import torch.nn as nn
import torchvision.models as models

def register_activation_hook(model, activations):
    """Dynamically register a hook on the penultimate layer to store activations."""

    def hook_fn(module, input, output):
        activations["penultimate"].append(output.detach().cpu().numpy())

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):  
        # For CNN-6, VGG-16 (Sequential classifiers)
        penultimate_layer_index = len(model.classifier) - 2  # Always take the second-to-last layer
        hook_handle = model.classifier[penultimate_layer_index].register_forward_hook(hook_fn)

    elif hasattr(model, "fc") and isinstance(model.fc, nn.Sequential):  
        # For ResNet-18 (Sequential FC layers)
        penultimate_layer_index = len(model.fc) - 2  # Always take the second-to-last layer
        hook_handle = model.fc[penultimate_layer_index].register_forward_hook(hook_fn)

    else:
        raise ValueError("Hook could not be registered. Model structure unknown.")

    return hook_handle
