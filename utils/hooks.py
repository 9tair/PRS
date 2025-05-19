import torch
import torch.nn as nn          # not strictly needed but nice for type hints

def register_activation_hook(
        model: nn.Module,
        activations: dict,
        model_name: str,
        dataset_name: str,
        batch_size: int,
        logger):
    """
    Forward–hook that collects the penultimate‑layer activations.

    The hook now:
        • returns immediately when activations["skip_batch"] is True
        • NEVER captures a brand‑new dict (we pass the same object for
          the whole run)
    """

    def hook_fn(module, inp, out):
        # ----- early‑exit guards -----------------------------------------
        if activations.get("skip_batch", False):
            return
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.warning(
                f"NaN/Inf in activations | {model_name=} {dataset_name=} "
                f"{batch_size=}"
            )
            activations["skip_batch"] = True
            return

        # ----- flatten & store -------------------------------------------
        if out.dim() > 2:
            out = torch.flatten(out, 1)

        out_detached = out.detach()          # still on CUDA
        activations["current"] = out_detached   # used immediately for the loss

        # store a *CPU* copy for later concatenation / MR computation
        activations["penultimate"].append(out_detached.cpu())

    # ----------------------------------------------------------------------
    if model_name == "VGG16":
        handle = model.classifier[3].register_forward_hook(hook_fn)
        logger.info("Hook registered at VGG16 classifier[3] (penultimate FC).")

    elif model_name == "ResNet18":
        handle = model.layer4[-1].register_forward_hook(hook_fn)
        logger.info("Hook registered at ResNet18 layer4[-1].")

    elif model_name == "CNN-6":
        handle = model.classifier[-2].register_forward_hook(hook_fn)
        logger.info("Hook registered at CNN‑6 penultimate FC.")

    else:
        raise ValueError(f"Unknown model structure: {model_name}")

    return handle