import sys
import os
import re
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

# Import project-specific modules
from config import config # Assuming config.py exists and has 'device'
from utils.adversarial_attacks import fgsm_attack, bim_attack, pgd_attack, cw_attack, autoattack
from utils import get_datasets
from models.model_factory import get_model

# --- Helper: Model Normalizer Wrapper for AutoAttack ---
class ModelNormalizerWrapper(torch.nn.Module):
    def __init__(self, model_to_wrap, mean, std):
        super().__init__()
        self.model_to_wrap = model_to_wrap
        # Ensure mean and std are correctly shaped for broadcasting (e.g., (1, C, 1, 1))
        self.mean = mean.view(1, -1, 1, 1) if mean.ndim == 1 else mean
        self.std = std.view(1, -1, 1, 1) if std.ndim == 1 else std

    def forward(self, x): # x is assumed to be [0,1]
        x_normalized = (x - self.mean) / self.std
        return self.model_to_wrap(x_normalized)

def get_dataset_normalization_params(dataset_name, device="cuda"):
    """Returns mean and std tensors for a given dataset."""
    if dataset_name == "CIFAR10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)
        std = torch.tensor([0.247, 0.243, 0.261], device=device)
    elif dataset_name == "MNIST" or dataset_name == "F-MNIST":
        mean = torch.tensor([0.1307], device=device)
        std = torch.tensor([0.3081], device=device)
    else: # Default, assuming [0,1] -> [-1,1] if not specified
        mean = torch.tensor([0.5, 0.5, 0.5], device=device)
        std = torch.tensor([0.5, 0.5, 0.5], device=device)
        print(f"Warning: Using default normalization for dataset {dataset_name}")
    return mean, std


def get_input_channels_from_dataset(dataset_name): # Unchanged from original
    return 1 if dataset_name in ["MNIST", "F-MNIST"] else 3

def load_model_from_folder(folder, device="cuda"):
    """Loads a model and its metadata from a given folder."""
    print(f"Attempting to load model from folder: {folder}")
    parent_folder = os.path.dirname(folder)
    if not parent_folder or parent_folder == folder:
        parent_folder_name = os.path.basename(folder)
    else:
        parent_folder_name = os.path.basename(parent_folder)

    model_name_tag = None

    if parent_folder_name.upper().startswith("CNN-6"): model_name_tag = "CNN-6"
    elif parent_folder_name.upper().startswith("VGG16"): model_name_tag = "VGG16"
    elif parent_folder_name.upper().startswith("RESNET18"): model_name_tag = "ResNet18"
    
    if model_name_tag is None:
        for file in os.listdir(folder):
            if file.endswith((".pth", ".pt")):
                file_lower = file.lower()
                if "cnn-6" in file_lower or "cnn6" in file_lower : model_name_tag = "CNN-6"
                elif "vgg16" in file_lower: model_name_tag = "VGG16"
                elif "resnet18" in file_lower: model_name_tag = "ResNet18"
                if model_name_tag: break
    
    if model_name_tag is None:
        folder_basename_upper = os.path.basename(folder).upper()
        if "CNN-6" in folder_basename_upper: model_name_tag = "CNN-6"
        elif "VGG16" in folder_basename_upper: model_name_tag = "VGG16"
        elif "RESNET18" in folder_basename_upper: model_name_tag = "ResNet18"
        
    if model_name_tag is None:
        raise ValueError(f"Could not infer model type (CNN-6, VGG16, ResNet18) from folder path: {folder}. Searched in parent '{parent_folder_name}' and files within.")

    print(f"Identified model type: {model_name_tag} from {folder}")
    dataset_name = extract_dataset_name(folder) or "CIFAR10"
    input_channels = get_input_channels_from_dataset(dataset_name)
    
    # Determine expected number of classes based on dataset
    # THIS IS THE SECTION TO DETERMINE NUM_CLASSES
    if dataset_name in ["CIFAR10", "MNIST", "F-MNIST"]:
        expected_num_classes = 10
    elif dataset_name == "ImageNet": # Example for future extension
        expected_num_classes = 1000
    else:
        print(f"Warning: Unknown dataset '{dataset_name}'. Assuming 10 output classes.")
        expected_num_classes = 10
        
    # Call your get_model without num_classes
    model = get_model(model_name_tag, input_channels=input_channels)

    # ---- POST-INSTANTIATION MODIFICATION FOR NUM_CLASSES (IF NEEDED) ----
    # Check if the model's output layer matches expected_num_classes
    # This part is heuristic and might need adjustment based on your specific model architectures
    if model_name_tag == "VGG16":
        # Your VGG16 get_model already sets the last layer to 10 outputs
        # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, expected_num_classes)
        # So, if expected_num_classes is not 10, and your get_model hardcodes it, this check is important.
        if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential) and \
           len(model.classifier) > 0 and isinstance(model.classifier[-1], torch.nn.Linear):
            if model.classifier[-1].out_features != expected_num_classes:
                print(f"Warning: VGG16 last layer output ({model.classifier[-1].out_features}) "
                      f"differs from expected ({expected_num_classes}) for dataset {dataset_name}. "
                      f"Your get_model might hardcode it. This script assumes it matches.")
                # If you absolutely needed to change it and can't modify get_model, you'd do it here.
                # model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, expected_num_classes)
        else:
            print(f"Warning: Could not inspect VGG16 classifier's last layer for num_classes matching.")

    elif model_name_tag == "ResNet18":
        # Your ResNet18 get_model already sets model.fc to 10 outputs
        # model.fc = nn.Linear(model.fc.in_features, expected_num_classes)
        if hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear):
            if model.fc.out_features != expected_num_classes:
                print(f"Warning: ResNet18 fc layer output ({model.fc.out_features}) "
                      f"differs from expected ({expected_num_classes}) for dataset {dataset_name}. "
                      f"Your get_model might hardcode it. This script assumes it matches.")
                # model.fc = torch.nn.Linear(model.fc.in_features, expected_num_classes)
        else:
            print(f"Warning: Could not inspect ResNet18 fc layer for num_classes matching.")

    elif model_name_tag == "CNN-6": # CustomCNN
        # For CustomCNN, we assume its definition correctly handles the number of classes
        # or that it also defaults to 10 and matches expected_num_classes.
        # If CustomCNN has a final linear layer named, e.g., 'fc' or 'output_layer', you could check it:
        # Example: if hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear):
        #              if model.fc.out_features != expected_num_classes: print("Warning...")
        pass # Assuming CustomCNN is correctly configured for 10 classes by default for these datasets

    # --- End of post-instantiation modification ---

    checkpoint_file = None
    preferred_files = ["best_model.pth", "model.pth", "checkpoint.pth", "final_model.pth"]
    for fname in preferred_files:
        if os.path.exists(os.path.join(folder, fname)):
            checkpoint_file = os.path.join(folder, fname)
            print(f"Found preferred checkpoint: {checkpoint_file}")
            break
    if checkpoint_file is None:
        for file in os.listdir(folder):
            if (file.endswith(".pth") or file.endswith(".pt")) and "optimizer" not in file.lower():
                checkpoint_file = os.path.join(folder, file)
                print(f"Found fallback checkpoint: {checkpoint_file}")
                break
    if checkpoint_file is None:
        raise FileNotFoundError(f"No model checkpoint (.pth or .pt) found in {folder}.")

    print(f"Loading weights from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    state_dict_to_load = None
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint: state_dict_to_load = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: state_dict_to_load = checkpoint['state_dict']
        elif 'model' in checkpoint: state_dict_to_load = checkpoint['model']
        else: state_dict_to_load = checkpoint
    else:
        state_dict_to_load = checkpoint

    is_parallel_model = all(k.startswith('module.') for k in state_dict_to_load.keys())
    if is_parallel_model:
        print("Removing 'module.' prefix from state_dict keys.")
        state_dict_to_load = {k[7:]: v for k, v in state_dict_to_load.items()}

    try:
        model.load_state_dict(state_dict_to_load)
    except RuntimeError as e:
        print(f"RuntimeError loading state_dict: {e}")
        print("Attempting to load with strict=False...")
        try:
            model.load_state_dict(state_dict_to_load, strict=False)
            print("Loaded with strict=False. Some keys might have been missing or unexpected.")
        except Exception as e2:
            print(f"Failed to load with strict=False as well: {e2}")
            print("Model's expected keys:")
            for k_model in model.state_dict().keys(): print(f"  - {k_model}")
            print("Checkpoint's keys:")
            for k_ckpt in state_dict_to_load.keys(): print(f"  - {k_ckpt}")
            raise e

    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    print("Testing model with random input...")
    dummy_input_shape = (1, input_channels, 32, 32) if input_channels == 3 else (1, input_channels, 28, 28)
    dummy_input = torch.randn(*dummy_input_shape).to(device)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Model output shape: {output.shape}. Expected num_classes for dataset: {expected_num_classes}")
        # Verify that the loaded model's output dimension matches the expected_num_classes
        if output.shape[1] != expected_num_classes:
            print(f"CRITICAL WARNING: Loaded model's output features ({output.shape[1]}) "
                  f"do NOT match expected num_classes ({expected_num_classes}) for dataset '{dataset_name}'. "
                  f"This will lead to errors or incorrect evaluations.")
            # You might want to raise an error here if it's critical:
            # raise ValueError("Model output dimension mismatch with dataset's expected classes.")
    except Exception as e:
        print(f"Error during dummy input test: {e}")

    model_desc = os.path.basename(folder)
    return model, model_desc

def check_model_accuracy(model, data_loader, device): # Unchanged
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Checking model accuracy", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f"Base model accuracy: {accuracy:.2f}%")
    return accuracy


def denormalize_tensor(tensor_normalized, mean_ds, std_ds):
    """Denormalizes a tensor given dataset mean and std."""
    # Ensure mean and std are tensors and on the same device as tensor_normalized
    mean_ds = mean_ds.to(tensor_normalized.device)
    std_ds = std_ds.to(tensor_normalized.device)

    # Reshape mean/std for broadcasting if they are 1D
    if mean_ds.ndim == 1:
        mean_ds = mean_ds.view(1, -1, 1, 1)
    if std_ds.ndim == 1:
        std_ds = std_ds.view(1, -1, 1, 1)
        
    return tensor_normalized * std_ds + mean_ds


def save_image_safely(tensor, filepath, dataset_name, is_normalized=True):
    """Safely saves an image. Handles denormalization if is_normalized=True."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tensor_to_save = tensor.clone().detach().cpu()

        if is_normalized:
            mean_ds_cpu, std_ds_cpu = get_dataset_normalization_params(dataset_name, device="cpu")
            tensor_to_save = denormalize_tensor(tensor_to_save, mean_ds_cpu, std_ds_cpu)
        
        tensor_to_save = torch.clamp(tensor_to_save, 0, 1) # Clip to [0,1] for saving
        save_image(tensor_to_save, filepath)
        # print(f"  Saved image to {filepath}") # Reduce verbosity
        return True
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")
        return False

def evaluate_robustness_with_samples(
    model_to_attack, data_loader, attack_fn_to_use, attack_parameters_base,
    epsilon_pixel_val,
    base_output_dir, current_attack_name, current_dataset_name, device_to_use,
    dataset_mean_for_scaling, dataset_std_for_scaling,
    num_samples_to_save=5,
    is_special_attack=False
):
    model_to_attack.eval()
    correct_adv = 0
    total_samples_processed = 0
    class_successful_attack_counts = defaultdict(int)
    class_misclassification_targets = defaultdict(lambda: defaultdict(int))
    samples_saved_this_run = 0 # This counter will now only track individual image saves

    attack_specific_dir = os.path.join(base_output_dir, current_attack_name)
    os.makedirs(attack_specific_dir, exist_ok=True)

    if is_special_attack or epsilon_pixel_val is None:
        run_subfolder_name = "fixed_params"
    else:
        run_subfolder_name = f"epsilon_pix_{epsilon_pixel_val:.4f}".replace('.', '_')
    
    current_run_output_dir = os.path.join(attack_specific_dir, run_subfolder_name)
    os.makedirs(current_run_output_dir, exist_ok=True)
    
    samples_visualization_dir = os.path.join(current_run_output_dir, "samples")
    os.makedirs(samples_visualization_dir, exist_ok=True)
    
    effective_attack_params = dict(attack_parameters_base)
    epsilon_for_attack_func = None

    if not is_special_attack and epsilon_pixel_val is not None:
        if torch.any(dataset_std_for_scaling == 0):
            raise ValueError("Dataset standard deviation contains zero, cannot scale epsilon/alpha.")
        std_reshaped = dataset_std_for_scaling.view(1, -1, 1, 1) if dataset_std_for_scaling.ndim == 1 else dataset_std_for_scaling
        epsilon_for_attack_func = epsilon_pixel_val / std_reshaped
        if "alpha" in effective_attack_params:
            alpha_pixel = effective_attack_params["alpha"]
            effective_attack_params["alpha"] = alpha_pixel / std_reshaped
    
    progress_bar_desc = f"Attacking: {current_attack_name}"
    if epsilon_pixel_val is not None and not is_special_attack:
        progress_bar_desc += f" (ε_pix={epsilon_pixel_val:.4f})"

    for batch_idx, (batch_inputs_from_loader, batch_labels) in enumerate(tqdm(data_loader, desc=progress_bar_desc, leave=False, unit="batch")):
        batch_labels = batch_labels.to(device_to_use)
        
        current_batch_inputs = None
        if current_attack_name == "AutoAttack_Linf": # Match the name used in main
            current_batch_inputs = denormalize_tensor(batch_inputs_from_loader.to(device_to_use), 
                                                      dataset_mean_for_scaling, 
                                                      dataset_std_for_scaling)
            current_batch_inputs = torch.clamp(current_batch_inputs, 0, 1)
        else:
            current_batch_inputs = batch_inputs_from_loader.to(device_to_use)
            
        with torch.no_grad():
            clean_outputs = model_to_attack(current_batch_inputs)
            _, clean_predictions = clean_outputs.max(1)

        adv_batch_inputs = None
        if current_attack_name == "AutoAttack_Linf":
            adv_batch_inputs = attack_fn_to_use(model_to_attack, current_batch_inputs, batch_labels, dataset_name=current_dataset_name).detach()
        elif current_attack_name == "CW_L2":
            adv_batch_inputs = attack_fn_to_use(model_to_attack, current_batch_inputs, batch_labels, **effective_attack_params).detach()
        else: 
            if epsilon_for_attack_func is None:
                raise ValueError(f"Epsilon for attack function is None for {current_attack_name}.")
            adv_batch_inputs = attack_fn_to_use(model_to_attack, current_batch_inputs, batch_labels, epsilon=epsilon_for_attack_func, **effective_attack_params).detach()

        with torch.no_grad():
            adv_outputs = model_to_attack(adv_batch_inputs)
            _, adv_predictions = adv_outputs.max(1)
            correct_adv += (adv_predictions == batch_labels).sum().item()
            total_samples_processed += batch_labels.size(0)

        for i in range(len(batch_labels)):
            true_label_val = int(batch_labels[i].item())
            clean_pred_val = int(clean_predictions[i].item())
            adv_pred_val = int(adv_predictions[i].item())

            if clean_pred_val == true_label_val and adv_pred_val != true_label_val:
                class_successful_attack_counts[true_label_val] += 1
                class_misclassification_targets[adv_pred_val][f"from_{true_label_val}"] += 1

                if samples_saved_this_run < num_samples_to_save:
                    single_orig_img_tensor = current_batch_inputs[i].cpu()
                    single_adv_img_tensor = adv_batch_inputs[i].cpu()

                    if single_orig_img_tensor.ndim == 4 and single_orig_img_tensor.shape[0] == 1:
                        single_orig_img_tensor = single_orig_img_tensor.squeeze(0)
                    if single_adv_img_tensor.ndim == 4 and single_adv_img_tensor.shape[0] == 1:
                        single_adv_img_tensor = single_adv_img_tensor.squeeze(0)
                    
                    is_img_normalized_for_saving = (current_attack_name != "AutoAttack_Linf")
                    
                    orig_path = os.path.join(samples_visualization_dir, f"s_{batch_idx}_{i}_orig_true_{true_label_val}.png")
                    if save_image_safely(single_orig_img_tensor, orig_path, current_dataset_name, is_normalized=is_img_normalized_for_saving):
                        samples_saved_this_run +=1 # Increment only if original is saved (or make it more granular)


                    adv_path = os.path.join(samples_visualization_dir, f"s_{batch_idx}_{i}_adv_true_{true_label_val}_pred_{adv_pred_val}.png")
                    save_image_safely(single_adv_img_tensor, adv_path, current_dataset_name, is_normalized=is_img_normalized_for_saving)
                    
                    # ------------------------------------------------------------------
                    # START OF COMMENTED OUT MATPLOTLIB PLOTTING BLOCK
                    # ------------------------------------------------------------------
                    # try:
                    #     fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                        
                    #     img_orig_for_plot_display = single_orig_img_tensor 
                    #     img_adv_for_plot_display = single_adv_img_tensor   

                    #     if is_img_normalized_for_saving: # which is (current_attack_name != "AutoAttack_Linf")
                    #          mean_cpu, std_cpu = get_dataset_normalization_params(current_dataset_name, "cpu")
                    #          img_orig_for_plot_display = denormalize_tensor(img_orig_for_plot_display, mean_cpu, std_cpu)
                    #          img_adv_for_plot_display = denormalize_tensor(img_adv_for_plot_display, mean_cpu, std_cpu)
                        
                    #     img_orig_for_plot_display = torch.clamp(img_orig_for_plot_display, 0, 1)
                    #     img_adv_for_plot_display = torch.clamp(img_adv_for_plot_display, 0, 1)

                    #     cmap_val = 'gray' if get_input_channels_from_dataset(current_dataset_name) == 1 else None
                        
                    #     if img_orig_for_plot_display.shape[0] == 1: 
                    #         display_orig_np = img_orig_for_plot_display.squeeze(0).numpy()
                    #     elif img_orig_for_plot_display.shape[0] == 3: 
                    #         display_orig_np = img_orig_for_plot_display.permute(1, 2, 0).numpy()
                    #     else: 
                    #         raise ValueError(f"Unexpected channel size for original image: {img_orig_for_plot_display.shape}")

                    #     axes[0].imshow(display_orig_np, cmap=cmap_val)
                    #     axes[0].set_title(f"Original: Actual {true_label_val}")
                    #     axes[0].axis('off')

                    #     if img_adv_for_plot_display.shape[0] == 1: 
                    #         display_adv_np = img_adv_for_plot_display.squeeze(0).numpy()
                    #     elif img_adv_for_plot_display.shape[0] == 3: 
                    #         display_adv_np = img_adv_for_plot_display.permute(1, 2, 0).numpy()
                    #     else: 
                    #         raise ValueError(f"Unexpected channel size for adversarial image: {img_adv_for_plot_display.shape}")

                    #     axes[1].imshow(display_adv_np, cmap=cmap_val)
                    #     axes[1].set_title(f"Adv: Pred {adv_pred_val} (Actual {true_label_val})")
                    #     axes[1].axis('off')
                        
                    #     comp_path = os.path.join(samples_visualization_dir, f"z_comparison_{samples_saved_this_run}.png") # samples_saved_this_run was for comparison images
                    #     plt.savefig(comp_path)
                    #     plt.close(fig)
                    #     # samples_saved_this_run was incremented here, now it's tied to individual saves.
                    # except ValueError as ve:
                    #     print(f"  Skipping comparison image due to shape/value error: {ve}")
                    # except Exception as e_plot:
                    #     print(f"  Error creating/saving comparison image (now commented out): {e_plot}")
                    #     # import traceback # Keep for debugging if you uncomment
                    #     # traceback.print_exc()
                    # ------------------------------------------------------------------
                    # END OF COMMENTED OUT MATPLOTLIB PLOTTING BLOCK
                    # ------------------------------------------------------------------

    adv_accuracy = 100.0 * correct_adv / total_samples_processed if total_samples_processed > 0 else 0.0
    
    try:
        _model_for_shape_check = model_to_attack.model_to_wrap if hasattr(model_to_attack, 'model_to_wrap') else model_to_attack
        num_actual_classes = _model_for_shape_check(torch.randn(1, get_input_channels_from_dataset(current_dataset_name), 32 if get_input_channels_from_dataset(current_dataset_name)==3 else 28, 32 if get_input_channels_from_dataset(current_dataset_name)==3 else 28).to(device_to_use)).shape[1]
    except Exception:
        num_actual_classes = 10

    stats_to_save = {
        "adversarial_accuracy": adv_accuracy,
        "attack_type": current_attack_name,
        "total_correct_adv": correct_adv,
        "total_samples": total_samples_processed,
        "successful_attacks_per_class": dict(sorted(class_successful_attack_counts.items(), key=lambda item: item[1], reverse=True)),
        "misclassification_targets_per_class": {
            str(cls_idx): dict(sorted(class_misclassification_targets[cls_idx].items(), key=lambda item: item[1], reverse=True))
            for cls_idx in range(num_actual_classes)
        }
    }
    if epsilon_pixel_val is not None and not is_special_attack:
        stats_to_save["epsilon_pixel"] = epsilon_pixel_val
    
    stats_file_path = os.path.join(current_run_output_dir, "run_statistics.json")
    with open(stats_file_path, "w") as f:
        json.dump(stats_to_save, f, indent=4)
    print(f"  Saved statistics for {current_attack_name} (ε_pix={epsilon_pixel_val if epsilon_pixel_val is not None else 'N/A'}) to {stats_file_path}")
    return stats_to_save

def extract_dataset_name(folder_path): # Unchanged
    """Extracts the dataset name from the folder path."""
    # More robust regex, case-insensitive for dataset names
    match = re.search(r"(CIFAR10|MNIST|F[-_]?MNIST)", folder_path, re.IGNORECASE)
    if match:
        name = match.group(0).upper()
        if name in ["F-MNIST", "FMNIST", "F_MNIST"]:
            return "F-MNIST" # Standardize to F-MNIST
        return name
    return None

def main():
    # --- Define Argument Parser FIRST ---
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness of a model.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing the model checkpoint.")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.01, 8/255, 0.05, 0.1],
                        help="Epsilon values (for [0,1] pixel space) for L-inf attacks. Default: 0.01, 8/255, 0.05, 0.1")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for data loaders.") # Default was 128, ensure it's what you want
    parser.add_argument("--num-save-samples", type=int, default=5, help="Number of adversarial sample images to save per run.")

    # --- THEN Parse Arguments ---
    args = parser.parse_args()

    # --- Device Setup ---
    use_cpu = args.cpu or not torch.cuda.is_available()
    # Assuming config is imported and available
    # from config import config # Make sure this import is at the top of the file if not already
    device = torch.device("cpu" if use_cpu else config.get("device", "cuda")) # Ensure 'config' is defined
    print(f"Using device: {device}")

    # --- Load Model ---
    try:
        # Ensure load_model_from_folder and extract_dataset_name are defined before this call
        model, model_description_name = load_model_from_folder(args.folder, device)
    except Exception as e:
        print(f"FATAL: Could not load model from {args.folder}. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Dataset and Normalization Parameters ---
    current_dataset_name = extract_dataset_name(args.folder)
    if current_dataset_name is None:
        print(f"Warning: Could not automatically determine dataset name from path '{args.folder}'. Defaulting to CIFAR10.")
        current_dataset_name = "CIFAR10"
    print(f"Determined dataset: {current_dataset_name}")

    # Ensure get_dataset_normalization_params is defined
    dataset_mean_vals, dataset_std_vals = get_dataset_normalization_params(current_dataset_name, device)

    # --- DataLoaders ---
    print(f"Loading '{current_dataset_name}' dataset (will be normalized by get_datasets)...")
    try:
        # Ensure get_datasets is defined and imported correctly
        # from utils import get_datasets # Example import
        train_dataset, test_dataset, _ = get_datasets(current_dataset_name)
    except Exception as e_data:
        print(f"FATAL: Could not load datasets using get_datasets('{current_dataset_name}'). Error: {e_data}")
        return
        
    test_loader_for_all_evals = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=not use_cpu
    )

    # --- Base Model Accuracy Check ---
    print("\nChecking base model accuracy on (normalized) test set...")
    # Ensure check_model_accuracy is defined
    base_model_test_accuracy = check_model_accuracy(model, test_loader_for_all_evals, device)

    if base_model_test_accuracy < 10.0:
        print(f"WARNING: Base model accuracy ({base_model_test_accuracy:.2f}%) is very low. Results might not be meaningful.")
        if input("Proceed with adversarial evaluation? (y/n): ").lower() != 'y':
            print("Exiting.")
            return
    
    # --- Output Directory Setup ---
    sanitized_model_desc = re.sub(r'[^\w\-_\. ]', '_', model_description_name)
    main_output_folder_name = f"robustness_eval_{sanitized_model_desc}_{current_dataset_name}"
    main_output_path = os.path.abspath(os.path.join(os.path.dirname(args.folder), main_output_folder_name))
    test_set_output_path = os.path.join(main_output_path, "test_set_results")
    os.makedirs(test_set_output_path, exist_ok=True)
    print(f"Results will be saved in: {test_set_output_path}")

    # --- Attack Configurations ---
    L_INF_ATTACKS_CONFIG = {
        "FGSM": {"attack_fn": fgsm_attack, "params": {}}, # Ensure fgsm_attack etc. are imported
        "BIM-10": {"attack_fn": bim_attack, "params": {"alpha": 0.01, "num_iter": 10}},
        "PGD-20": {"attack_fn": pgd_attack, "params": {"alpha": (2/255), "num_iter": 20, "restarts": 1}},
    }
    PIXEL_SPACE_EPSILONS = args.epsilon
    all_run_results_summary = {}

    # --- Evaluate L-infinity Attacks (FGSM, BIM, PGD) ---
    for attack_name_key, attack_config_data in L_INF_ATTACKS_CONFIG.items():
        print(f"\n{'='*25} EVALUATING: {attack_name_key} {'='*25}")
        results_for_this_attack = {}
        for current_eps_pixel in PIXEL_SPACE_EPSILONS:
            print(f"\n--- Running {attack_name_key} with pixel_epsilon = {current_eps_pixel:.5f} ---")
            current_attack_params = dict(attack_config_data["params"])
            # Ensure evaluate_robustness_with_samples is defined
            test_run_stats = evaluate_robustness_with_samples(
                model, test_loader_for_all_evals,
                attack_config_data["attack_fn"], current_attack_params,
                current_eps_pixel, 
                test_set_output_path, attack_name_key, current_dataset_name, device,
                dataset_mean_vals, dataset_std_vals,
                num_samples_to_save=args.num_save_samples,
                is_special_attack=False
            )
            results_for_this_attack[f"{current_eps_pixel:.5f}"] = {"test_accuracy": test_run_stats["adversarial_accuracy"]}
        all_run_results_summary[attack_name_key] = results_for_this_attack

    # --- Evaluate CW (L2) Attack ---
    print(f"\n{'='*25} EVALUATING: CW Attack (L2) {'='*25}")
    cw_attack_params = {"c": 1e-2, "kappa": 0, "max_iter": 100, "lr": 0.01} # Ensure cw_attack is imported
    test_run_stats_cw = evaluate_robustness_with_samples(
        model, test_loader_for_all_evals,
        cw_attack, cw_attack_params,
        epsilon_pixel_val=None,
        base_output_dir=test_set_output_path, current_attack_name="CW_L2", 
        current_dataset_name=current_dataset_name, device_to_use=device,
        dataset_mean_for_scaling=dataset_mean_vals, dataset_std_for_scaling=dataset_std_vals,
        num_samples_to_save=args.num_save_samples,
        is_special_attack=True
    )
    all_run_results_summary["CW_L2"] = {"test_accuracy": test_run_stats_cw["adversarial_accuracy"]}

    # --- Evaluate AutoAttack (L-inf) ---
    print(f"\n{'='*25} EVALUATING: AutoAttack (L-inf) {'='*25}")
    # Ensure ModelNormalizerWrapper is defined
    model_wrapped_for_autoattack = ModelNormalizerWrapper(model, dataset_mean_vals.to(device), dataset_std_vals.to(device)).to(device)
    model_wrapped_for_autoattack.eval()
    # Ensure autoattack is imported
    test_run_stats_aa = evaluate_robustness_with_samples(
        model_wrapped_for_autoattack,
        test_loader_for_all_evals,
        autoattack, {},
        epsilon_pixel_val=None,
        base_output_dir=test_set_output_path, current_attack_name="AutoAttack_Linf", 
        current_dataset_name=current_dataset_name, device_to_use=device,
        dataset_mean_for_scaling=dataset_mean_vals, dataset_std_for_scaling=dataset_std_vals,
        num_samples_to_save=args.num_save_samples,
        is_special_attack=True
    )
    all_run_results_summary["AutoAttack_Linf"] = {"test_accuracy": test_run_stats_aa["adversarial_accuracy"]}
    
    # --- Final Summary Report ---
    try:
        # Ensure get_input_channels_from_dataset is defined
        num_model_classes = model(torch.randn(1, get_input_channels_from_dataset(current_dataset_name), 32 if get_input_channels_from_dataset(current_dataset_name)==3 else 28, 32 if get_input_channels_from_dataset(current_dataset_name)==3 else 28).to(device)).shape[1]
    except Exception:
        num_model_classes = 10

    dataset_class_labels = {}
    if current_dataset_name == "CIFAR10": dataset_class_labels = {str(i): lbl for i, lbl in enumerate(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])}
    elif current_dataset_name == "MNIST": dataset_class_labels = {str(i): str(i) for i in range(10)}
    elif current_dataset_name == "F-MNIST": dataset_class_labels = {str(i): lbl for i, lbl in enumerate(["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])}
    else: dataset_class_labels = {str(i): f"Class {i}" for i in range(num_model_classes)}

    final_summary_data = {
        "model_description": model_description_name,
        "model_folder": args.folder,
        "dataset_evaluated": current_dataset_name,
        "dataset_class_labels": dataset_class_labels,
        "base_model_test_accuracy_percent": base_model_test_accuracy,
        "adversarial_evaluation_results": all_run_results_summary,
        "evaluation_device": str(device),
        "epsilons_tested_pixel_space": PIXEL_SPACE_EPSILONS
    }
    
    summary_report_filepath = os.path.join(main_output_path, "evaluation_summary_report.json")
    with open(summary_report_filepath, "w") as f:
        json.dump(final_summary_data, f, indent=4)

    print(f"\n{'='*20} EVALUATION COMPLETE {'='*20}")
    print(f"All results and samples saved in main directory: {main_output_path}")
    print(f"Overall summary report saved to: {summary_report_filepath}")

if __name__ == "__main__":
    # Ensure config is imported for device configuration if not already imported globally
    from config import config # Place this here or ensure it's at the top of the script
    main()