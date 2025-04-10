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
from config import config
from utils import fgsm_attack, bim_attack, pgd_attack, get_datasets, cw_attack, autoattack
from models.model_factory import get_model

def get_input_channels_from_dataset(dataset_name):
    return 1 if dataset_name in ["MNIST", "FMNIST"] else 3

def load_model_from_folder(folder, device="cuda"):
    """Loads a model and its metadata from a given folder."""
    # Print all files in the folder for debugging
    print(f"Files in {folder}:")
    for file in os.listdir(folder):
        print(f"  - {file}")
    """Loads a model and its metadata from a given folder."""
    # Extract the base folder name to determine model type
    folder_name = os.path.basename(os.path.dirname(folder))  # Get parent folder name
    model_name = None

    # Identify the model type based on folder naming
    if folder_name.startswith("CNN-6"):
        model_name = "CNN-6"
    elif folder_name.startswith("VGG16"):
        model_name = "VGG16"
    elif folder_name.startswith("ResNet18"):
        model_name = "ResNet18"
    else:
        # Try to infer model type from checkpoint file name
        for file in os.listdir(folder):
            if file.endswith(".pth") or file.endswith(".pt"):
                if "cnn" in file.lower():
                    model_name = "CNN-6"
                elif "vgg" in file.lower():
                    model_name = "VGG16"
                elif "resnet" in file.lower():
                    model_name = "ResNet18"
                break
        
        if model_name is None:
            raise ValueError(f"Unsupported model path: {folder}. Ensure the model folder contains a supported model.")

    print(f"Loading model: {model_name} from {folder}")

    # Determine input channels based on dataset
    dataset_name = extract_dataset_name(folder) or "CIFAR10"
    _, _, input_channels = get_datasets(dataset_name)

    # Create model architecture
    model = get_model(model_name, input_channels=input_channels)

    # Find model checkpoint file (avoiding optimizer.pth)
    checkpoint_file = None
    for file in os.listdir(folder):
        if (file.endswith(".pth") or file.endswith(".pt")) and not file.startswith("optimizer"):
            checkpoint_file = os.path.join(folder, file)
            break
    
    # If no specific model file found, look for checkpoint.pth
    if checkpoint_file is None:
        if os.path.exists(os.path.join(folder, "checkpoint.pth")):
            checkpoint_file = os.path.join(folder, "checkpoint.pth")
        elif os.path.exists(os.path.join(folder, "model.pth")):
            checkpoint_file = os.path.join(folder, "model.pth")
        elif os.path.exists(os.path.join(folder, "best_model.pth")):
            checkpoint_file = os.path.join(folder, "best_model.pth")
    
    if checkpoint_file is None:
        raise FileNotFoundError(f"No model checkpoint found in {folder}. Please ensure there's a .pth file containing model weights.")
    
    # Load model weights
    print(f"Loading weights from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Check if checkpoint has 'model_state_dict' key or is the state dict itself
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Model architecture:\n{model}")
    
    # Test if model is loaded properly
    print("Testing model with random input...")
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Adjust size based on your dataset
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    
    model_desc = os.path.basename(folder)  # Store folder name as model description
    return model, model_desc

def check_model_accuracy(model, data_loader, device):
    """Tests base model accuracy to verify it's working properly."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Checking model accuracy"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Base model accuracy: {accuracy:.2f}%")
    
    return accuracy

def denormalize(tensor, mean, std):
    # Get the number of channels from the mean tensor
    num_channels = mean.size(0)
    
    # Reshape according to number of channels
    if num_channels == 1:  # For grayscale (MNIST, FMNIST)
        mean = mean.view(1, 1, 1)
        std = std.view(1, 1, 1)
    else:  # For RGB (CIFAR10, etc.)
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
        
    return tensor * std + mean

def save_image_safely(tensor, filepath, dataset="CIFAR10"):
    """Safely saves an image with proper error handling."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Handle the case where tensor is a list
        if isinstance(tensor, list):
            tensor = torch.stack(tensor)  # Convert list of tensors to a single tensor
        
        # Ensure tensor is on CPU and detached from computation graph
        tensor_to_save = tensor.clone().detach().cpu()

        # Define mean and std based on dataset
        if dataset == "CIFAR10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.247, 0.243, 0.261])
        elif dataset == "MNIST" or dataset == "FMNIST":
            mean = torch.tensor([0.1307])
            std = torch.tensor([0.3081])
        else:
            mean = torch.tensor([0.5, 0.5, 0.5])  # Default to generic normalization
            std = torch.tensor([0.5, 0.5, 0.5])

        # Reshape mean and std for broadcasting
        if dataset == "MNIST" or dataset == "FMNIST":
            mean = mean.view(1, 1, 1)
            std = std.view(1, 1, 1)
        else:
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)

        # Denormalize
        tensor_to_save = denormalize(tensor_to_save, mean, std)
        
        # Clip values to ensure they are in range [0,1]
        tensor_to_save = torch.clamp(tensor_to_save, 0, 1)

        save_image(tensor_to_save, filepath)
        print(f"  Saved image to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")
        return False


def evaluate_robustness_with_samples(model, data_loader, attack_fn, attack_params, epsilon, 
                                    output_dir, attack_name, dataset_name, device="cuda", num_samples_to_save=5):
    """Evaluates model robustness, collects statistics, and saves adversarial examples."""
    model.to(device).eval()
    correct = 0
    total = 0
    class_success_count = defaultdict(int)  # Count of successfully attacked samples per true class
    class_target_count = defaultdict(lambda: defaultdict(int))  # Track misclassification distributions
    
    # Track samples to save (limit to num_samples_to_save per attack)
    samples_saved = 0
    
    # Create directory for this attack
    attack_dir = os.path.abspath(os.path.join(output_dir, attack_name))
    os.makedirs(attack_dir, exist_ok=True)
    
    # Create directory for epsilon
    epsilon_dir = os.path.abspath(os.path.join(attack_dir, f"epsilon_{epsilon}"))
    os.makedirs(epsilon_dir, exist_ok=True)
    
    # Create samples directory
    samples_dir = os.path.abspath(os.path.join(epsilon_dir, "samples"))
    os.makedirs(samples_dir, exist_ok=True)
    
    print(f"Will save samples to: {samples_dir}")
    
    # Define dataset-specific normalization parameters
    if "CIFAR10" in output_dir:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1).to(device)
    elif "MNIST" in output_dir or "FMNIST" in output_dir:
        mean = torch.tensor([0.1307]).view(1, 1, 1, 1).to(device)
        std = torch.tensor([0.3081]).view(1, 1, 1, 1).to(device)
    else:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)  # Default
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"Running {attack_name} (ε={epsilon})")):
        inputs, labels = inputs.to(device), labels.to(device)
        original_inputs = inputs.clone().detach()

        # Clean prediction
        with torch.no_grad():
            clean_outputs = model(inputs)
            _, clean_predicted = clean_outputs.max(1)

        # Generate adversarial examples
        if attack_name == "AutoAttack":
            adv_inputs = attack_fn(model, inputs.clone(), labels, dataset_name=dataset_name).detach()
        elif attack_name == "CW":
            adv_inputs = attack_fn(model, inputs.clone(), labels, **attack_params).detach()
        else:
            adv_inputs = attack_fn(model, inputs.clone(), labels, epsilon=epsilon, **attack_params).detach()

        # Adversarial prediction
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Collect misclassification statistics
        for i in range(len(labels)):
            true_label = int(labels[i].cpu().numpy())  
            clean_pred = int(clean_predicted[i].cpu().numpy())  
            adv_pred = int(predicted[i].cpu().numpy())

            # Only count adversarial misclassification if it was originally classified correctly
            if clean_pred == true_label and adv_pred != true_label:
                class_success_count[true_label] += 1  # Increment count for successful attack
                
                # Track which class it was misclassified into
                class_target_count[adv_pred][f"from_{true_label}"] += 1

                # Save sample images (original and adversarial) if needed
                if samples_saved < num_samples_to_save:
                    success = True

                    # Original image
                    orig_path = os.path.join(samples_dir, f"sample_{batch_idx}_{i}_original_class_{true_label}.png")
                    success &= save_image_safely(inputs[i].cpu(), orig_path)

                    # Adversarial image
                    adv_path = os.path.join(samples_dir, f"sample_{batch_idx}_{i}_adversarial_class_{true_label}_to_{adv_pred}.png")
                    success &= save_image_safely(adv_inputs[i].cpu(), adv_path)

                    # Create comparison image
                    if success:
                        # Use this to replace the comparison image creation section in evaluate_robustness_with_samples function
                        try:
                            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                            # Ensure tensors are properly handled
                            orig_tensor = inputs[i].clone().detach().cpu()
                            adv_tensor = adv_inputs[i].clone().detach().cpu()

                            # Create device-independent mean and std tensors
                            if "CIFAR10" in dataset_name:
                                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                                std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
                            elif "MNIST" in dataset_name or "FMNIST" in dataset_name:
                                mean = torch.tensor([0.1307]).view(1, 1, 1)
                                std = torch.tensor([0.3081]).view(1, 1, 1)
                            else:
                                mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                                std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

                            # Extract original and adversarial images
                            original_img = denormalize(orig_tensor, mean, std).numpy()
                            adversarial_img = denormalize(adv_tensor, mean, std).numpy()

                            # Handle channel dimensions based on dataset
                            if dataset_name == "MNIST" or dataset_name == "FMNIST":
                                original_img = original_img.squeeze(0)  # Remove channel dimension for grayscale
                                adversarial_img = adversarial_img.squeeze(0)
                            else:
                                original_img = original_img.transpose(1, 2, 0)  # Change from CxHxW to HxWxC for color
                                adversarial_img = adversarial_img.transpose(1, 2, 0)

                            # Ensure pixel values are in [0,1] range
                            original_img = np.clip(original_img, 0, 1)
                            adversarial_img = np.clip(adversarial_img, 0, 1)

                            # Plot original image
                            axes[0].imshow(original_img, cmap='gray' if dataset_name in ["MNIST", "FMNIST"] else None)
                            axes[0].set_title(f"Original: Class {true_label}")
                            axes[0].axis('off')

                            # Plot adversarial image
                            axes[1].imshow(adversarial_img, cmap='gray' if dataset_name in ["MNIST", "FMNIST"] else None)
                            axes[1].set_title(f"Adversarial: Predicted as {adv_pred}")
                            axes[1].axis('off')

                            # Save comparison image
                            comp_path = os.path.join(samples_dir, f"comparison_{samples_saved}.png")
                            plt.savefig(comp_path)
                            plt.close()

                            samples_saved += 1
                            print(f"Saved comparison {samples_saved}/{num_samples_to_save} to {samples_dir}")

                        except Exception as e:
                            print(f"Error creating comparison image: {e}")
                            import traceback
                            traceback.print_exc()  # Print full traceback for debugging
                            
    # Calculate accuracy
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    # Sort class_success_count in descending order
    sorted_success_count = dict(sorted(class_success_count.items(), key=lambda item: item[1], reverse=True))

    # Sort class_target_count:
    sorted_target_count = {}
    for class_label in range(10):  # Ensuring classes 0-9 are always present
        if class_label in class_target_count:
            sorted_target_count[str(class_label)] = dict(
                sorted(class_target_count[class_label].items(), key=lambda item: item[1], reverse=True)
            )
        else:
            sorted_target_count[str(class_label)] = {}  # Ensure all classes exist even if empty

    stats = {
        "accuracy": accuracy,
        "epsilon": epsilon,
        "attack_type": attack_name,
        "total_correct": correct,
        "total_samples": total,
        "class_success_count": sorted_success_count,
        "class_target_count": sorted_target_count  # Now sorted properly
    }

    # Save statistics
    stats_path = os.path.join(epsilon_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved statistics to {stats_path}")
    return stats

def extract_dataset_name(folder_path):
    """
    Extracts the dataset name from the folder path.
    Assumes dataset name follows a standard convention.
    """
    match = re.search(r"CIFAR10|MNIST|F[-_]?MNIST", folder_path)
    if match:
        name = match.group(0)
        if name in ["F-MNIST", "FMNIST", "F_MNIST"]:
            return "F-MNIST"
        return name

def main():
    """Main function to evaluate adversarial robustness for a single model."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness of a model.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the model folder.")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.01, 0.0313, 0.05, 0.1], 
                        help="Epsilon values to test (default: 0.01 0.0313 0.05 1.0)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--model-file", type=str, help="Specific model file to load (if not automatically detected)")
    args = parser.parse_args()

    # Determine device
    if args.cpu:
        device = "cpu"
    else:
        device = config["device"] if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load trained model
    try:
        model, model_desc = load_model_from_folder(args.folder, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative loading method...")
        
        # Try a more direct approach if the folder-based loading fails
        model_name = "CNN-6"  # Default model
        if "VGG" in args.folder:
            model_name = "VGG16"
        elif "ResNet" in args.folder:
            model_name = "ResNet18"
        
        print(f"Creating model architecture: {model_name}")
        dataset_name = extract_dataset_name(args.folder) or "CIFAR10"
        _, _, input_channels = get_datasets(dataset_name)
        model = get_model(model_name, input_channels=input_channels)

        # If specific model file is provided, use it
        if args.model_file:
            model_path = args.model_file
        else:
            print("Searching for model files in parent directories...")
            
            # Try to find model files in parent directories
            parent_dir = os.path.dirname(args.folder)
            model_path = None
            
            for root, dirs, files in os.walk(parent_dir, topdown=False):
                for file in files:
                    if file.endswith(".pth") and not file.startswith("optimizer"):
                        model_path = os.path.join(root, file)
                        print(f"Found model file: {model_path}")
                        break
                if model_path:
                    break
        
        if not model_path:
            raise FileNotFoundError("Could not find a model checkpoint file in the folder or parent directories")
        
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Try different ways to load the state dict
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Try to load directly, skipping "module." prefix if it exists
                # (this handles models trained with DataParallel)
                try:
                    model.load_state_dict(checkpoint)
                except:
                    # If that fails, try to remove "module." prefix
                    new_state_dict = {}
                    for k, v in checkpoint.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v
                        else:
                            new_state_dict[k] = v
                    model.load_state_dict(new_state_dict)
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Model keys:")
            for name, _ in model.named_parameters():
                print(f"  - {name}")
            
            print("\nCheckpoint keys:")
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    for k in checkpoint['model_state_dict'].keys():
                        print(f"  - {k}")
                elif 'state_dict' in checkpoint:
                    for k in checkpoint['state_dict'].keys():
                        print(f"  - {k}")
                else:
                    for k in checkpoint.keys():
                        print(f"  - {k}")
            
            raise e
        
        model = model.to(device)
        model.eval()
        model_desc = os.path.basename(args.folder)
    
    # Extract dataset name
    dataset_name = extract_dataset_name(args.folder)
    if dataset_name is None:
        print("Warning: Could not determine dataset name from path. Defaulting to CIFAR10.")
        dataset_name = "CIFAR10"
    
    print(f"Using dataset: {dataset_name}")
    
    # Load datasets
    train_dataset, test_dataset, _ = get_datasets(dataset_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Check base model accuracy to ensure it's loaded properly
    print("\nChecking base model accuracy on test set...")
    base_test_acc = check_model_accuracy(model, test_loader, device)
    
    print("\nChecking base model accuracy on train set...")
    base_train_acc = check_model_accuracy(model, train_loader, device)
    
    if base_test_acc < 10.0 or base_train_acc < 10.0:
        print("WARNING: Model accuracy is very low. The model might not be loaded correctly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    # Create output directories using absolute paths
    output_dir = os.path.abspath(os.path.join(args.folder, "attacks"))
    train_output_dir = os.path.abspath(os.path.join(output_dir, "train"))
    test_output_dir = os.path.abspath(os.path.join(output_dir, "test"))
    
    # Create directories
    for dir_path in [output_dir, train_output_dir, test_output_dir]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Attack configurations with all required parameters
    ATTACKS = {
    "FGSM": {"attack_fn": fgsm_attack, "params": {}},
    "BIM": {"attack_fn": bim_attack, "params": {"alpha": 0.01, "num_iter": 10}},
    "PGD-20": {"attack_fn": pgd_attack, "params": {"alpha": 0.007, "num_iter": 20}},
    "CW": {"attack_fn": cw_attack, "params": {"c": 1e-4, "kappa": 0, "max_iter": 1000, "lr": 0.01}},
    "AutoAttack": {"attack_fn": autoattack, "params": {}},
    }
    
    # Use provided epsilon values or defaults
    EPSILONS = args.epsilon
    
    # Store class mapping for meaningful labels in the dataset
    class_mapping = {}
    try:
        if dataset_name == "CIFAR10":
            class_mapping = {
                "0": "airplane", "1": "automobile", "2": "bird", "3": "cat", 
                "4": "deer", "5": "dog", "6": "frog", "7": "horse", 
                "8": "ship", "9": "truck"
            }
        elif dataset_name == "MNIST":
            class_mapping = {
                "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
                "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"
            }
        elif dataset_name == "FMNIST":
            class_mapping = {
                "0": "T-shirt/top", "1": "Trouser", "2": "Pullover", "3": "Dress", 
                "4": "Coat", "5": "Sandal", "6": "Shirt", "7": "Sneaker", 
                "8": "Bag", "9": "Ankle boot"
            }
    except Exception as e:
        print(f"Warning: Could not create class mapping: {e}")
    
    # Store all results for summary
    all_results = {}
    
    for attack_name, attack_data in ATTACKS.items():
        attack_results = {}
        
        if attack_name == "CW":
            # Run CW attack only once (ignoring ε)
            print(f"\n{'='*50}")
            print(f"Running {attack_name} (L2 attack without epsilon constraint)")
            print(f"{'='*50}")

            # Run evaluation on test set
            print(f"\nEvaluating {attack_name} on test set")
            test_stats = evaluate_robustness_with_samples(
                model, 
                test_loader,
                attack_data["attack_fn"], 
                attack_data["params"],  # No epsilon
                None,  # Pass None for ε
                test_output_dir,
                attack_name,
                dataset_name,
                device
            )

            # Run evaluation on train set
            print(f"\nEvaluating {attack_name} on train set")
            train_stats = evaluate_robustness_with_samples(
                model, 
                train_loader,
                attack_data["attack_fn"], 
                attack_data["params"], 
                None,  # Pass None for ε
                train_output_dir,
                attack_name,
                dataset_name,
                device
            )

            # Store accuracy for summary
            all_results[attack_name] = {
                "test_accuracy": test_stats["accuracy"],
                "train_accuracy": train_stats["accuracy"]
            }

        else:
            for epsilon in EPSILONS:
                print(f"\n{'='*50}")
                print(f"Running {attack_name} with epsilon={epsilon}")
                print(f"{'='*50}")

                # Run evaluation on test set
                print(f"\nEvaluating {attack_name} with epsilon={epsilon} on test set")
                test_stats = evaluate_robustness_with_samples(
                    model, 
                    test_loader,
                    attack_data["attack_fn"], 
                    attack_data["params"], 
                    epsilon, 
                    test_output_dir,
                    attack_name,
                    dataset_name,
                    device
                )

                # Run evaluation on train set
                print(f"\nEvaluating {attack_name} with epsilon={epsilon} on train set")
                train_stats = evaluate_robustness_with_samples(
                    model, 
                    train_loader,
                    attack_data["attack_fn"], 
                    attack_data["params"], 
                    epsilon, 
                    train_output_dir,
                    attack_name,
                    dataset_name,
                    device
                )

                # Store accuracy for summary
                all_results[attack_name] = all_results.get(attack_name, {})
                all_results[attack_name][str(epsilon)] = {
                    "test_accuracy": test_stats["accuracy"],
                    "train_accuracy": train_stats["accuracy"]
                }

    # Save summary of accuracies to a JSON file
    summary = {
        "model_name": model_desc,
        "dataset": dataset_name,
        "class_mapping": class_mapping,
        "base_accuracy": {
            "test": base_test_acc,
            "train": base_train_acc
        },
        "accuracy_results": all_results
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    
    def aggregate_stats(base_dir):
        all_stats = {}
        for attack_name in os.listdir(base_dir):
            attack_dir = os.path.join(base_dir, attack_name)
            if not os.path.isdir(attack_dir):
                continue
            for eps_folder in os.listdir(attack_dir):
                eps_dir = os.path.join(attack_dir, eps_folder)
                stats_file = os.path.join(eps_dir, "stats.json")
                if os.path.exists(stats_file):
                    with open(stats_file, "r") as f:
                        stats = json.load(f)
                    if attack_name not in all_stats:
                        all_stats[attack_name] = {}
                    all_stats[attack_name][eps_folder] = stats
        return all_stats

    # Aggregate test and train stats
    test_agg_stats = aggregate_stats(test_output_dir)
    train_agg_stats = aggregate_stats(train_output_dir)

    # Save to JSON
    with open(os.path.join(test_output_dir, "all_stats.json"), "w") as f:
        json.dump(test_agg_stats, f, indent=4)

    with open(os.path.join(train_output_dir, "all_stats.json"), "w") as f:
        json.dump(train_agg_stats, f, indent=4)

    print(f"Aggregated test stats saved to {os.path.join(test_output_dir, 'all_stats.json')}")
    print(f"Aggregated train stats saved to {os.path.join(train_output_dir, 'all_stats.json')}")
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAll results saved to {output_dir}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    
    main()