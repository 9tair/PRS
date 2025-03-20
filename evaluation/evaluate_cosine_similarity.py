import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.spatial.distance import cosine
from pathlib import Path

def extract_model_info(model_path):
    """
    Extract model name, dataset name, batch size, and epoch (if available) from the path.
    
    Args:
        model_path (str or Path): Path to the model directory
        
    Returns:
        Tuple: (model_name, dataset_name, batch_size, epoch_num)
    """
    model_path = Path(model_path)
    path_parts = str(model_path).split(os.sep)
    
    # Check if we're in an epoch folder
    epoch_num = None
    epoch_match = re.search(r"epoch_(\d+)", str(model_path))
    if epoch_match:
        epoch_num = int(epoch_match.group(1))
        # If we're in an epoch folder, get the parent directory for model info
        if "epoch_" in path_parts[-1]:
            model_dir = path_parts[-2]
        else:
            model_dir = path_parts[-1]
    else:
        model_dir = path_parts[-1]
    
    # Extract model parameters
    match = re.search(r"(.+?)_(.+?)_batch_(\d+)", model_dir)
    if not match:
        # If no match, use generic labels
        return "Unknown", "Unknown", "Unknown", epoch_num
    
    model_name, dataset_name, batch_size = match.groups()
    return model_name, dataset_name, int(batch_size), epoch_num

def find_final_layer_weights(model_data):
    """
    Find the final classification layer weights in the model.
    Supports various model architectures and naming conventions.
    
    Args:
        model_data (dict): Model state dictionary
        
    Returns:
        tuple: (final_layer_weights, key_name) or (None, None) if not found
    """
    # First, try common patterns for the final layer
    final_layer_patterns = [
        r"fc\.weight$",
        r"classifier\.weight$", 
        r"linear\.weight$",
        r"out_proj\.weight$",
        r"output.*\.weight$",
        r"head.*\.weight$"
    ]
    
    for pattern in final_layer_patterns:
        for key in model_data.keys():
            if re.search(pattern, key):
                return model_data[key].cpu().numpy(), key
    
    # If not found, look for nested classifiers (like in VGG or some CNN architectures)
    # Check for patterns like classifier.6.weight, where the highest number is likely the final layer
    classifier_layers = {}
    for key in model_data.keys():
        # Look for patterns like classifier.X.weight or fc.X.weight where X is a number
        nested_match = re.search(r"(classifier|fc|linear)\.(\d+)\.weight$", key)
        if nested_match:
            layer_num = int(nested_match.group(2))
            classifier_layers[layer_num] = key
    
    # If we found nested classifier layers, return the one with the highest number
    if classifier_layers:
        highest_layer = max(classifier_layers.keys())
        key = classifier_layers[highest_layer]
        return model_data[key].cpu().numpy(), key
    
    # If still not found, check for any key ending with .weight that might be a classification layer
    # Prioritize keys that contain 'classifier', 'fc', 'linear' and end with .weight
    potential_keys = [k for k in model_data.keys() if k.endswith('.weight') and 
                     ('classifier' in k or 'fc' in k or 'linear' in k)]
    
    if potential_keys:
        # Sort by complexity - assume the longer/more nested key is more likely the final layer
        potential_keys.sort(key=lambda x: x.count('.'), reverse=True)
        key = potential_keys[0]
        return model_data[key].cpu().numpy(), key
    
    return None, None

def compute_cosine_similarity(weight_matrix):
    """
    Compute cosine similarity between all class vectors in the final layer.
    
    Args:
        weight_matrix (numpy.ndarray): A 2D matrix where each row corresponds to the final-layer class vector.

    Returns:
        numpy.ndarray: Cosine similarity matrix of shape (num_classes, num_classes).
    """
    num_classes = weight_matrix.shape[0]
    similarity_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            similarity_matrix[i, j] = 1 - cosine(weight_matrix[i], weight_matrix[j])  # 1 - cosine distance

    return similarity_matrix

def visualize_cosine_similarity(similarity_matrix, save_path, model_info=None, layer_name=None):
    """
    Create and save a heatmap visualization of the cosine similarity matrix.

    Args:
        similarity_matrix (numpy.ndarray): Cosine similarity matrix.
        save_path (str): Path to save the heatmap image.
        model_info (tuple, optional): Tuple containing (model_name, dataset_name, batch_size, epoch_num)
        layer_name (str, optional): Name of the layer used for similarity computation
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    
    # Create title with model information if available
    if model_info:
        model_name, dataset_name, batch_size, epoch_num = model_info
        title = f"Cosine Similarity - {model_name} {dataset_name}, Batch {batch_size}"
        if epoch_num is not None:
            title += f", Epoch {epoch_num}"
        if layer_name:
            title += f"\nLayer: {layer_name}"
        plt.title(title)
    else:
        title = "Cosine Similarity of Final Layer Weights"
        if layer_name:
            title += f"\nLayer: {layer_name}"
        plt.title(title)
    
    plt.xlabel("Class Index")
    plt.ylabel("Class Index")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_cosine_similarity(model_folder):
    """
    Evaluate cosine similarity of the final classification layer in the model.
    Works with both direct model directories and epoch subdirectories.

    Args:
        model_folder (str): Path to the model's saved folder.
    """
    model_folder = Path(model_folder)
    
    # Extract model information from path
    model_info = extract_model_info(model_folder)
    model_name, dataset_name, batch_size, epoch_num = model_info
    
    print(f"Analyzing model: {model_name} {dataset_name}, Batch Size: {batch_size}" + 
            (f", Epoch {epoch_num}" if epoch_num is not None else ""))
    
    # Look for the model file (assuming it's a .pth or .pt file)
    model_files = list(model_folder.glob("*.pth")) + list(model_folder.glob("*.pt"))
    
    # If multiple model files exist, prioritize ones with names like "model.pth" or "weights.pth"
    if len(model_files) > 1:
        for preferred_name in ["model.pth", "weights.pth"]:
            preferred_files = [f for f in model_files if f.name == preferred_name]
            if preferred_files:
                model_files = preferred_files
                break
    
    if not model_files:
        print(f"No model checkpoint found in {model_folder}")
        return
    
    model_path = model_files[0]  # Use the first (or preferred) model file
    print(f"Using model file: {model_path}")
    
    try:
        model_data = torch.load(model_path, map_location=torch.device("cpu"))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Handle both state_dict and direct weight formats
    if not isinstance(model_data, dict):
        print("Model file does not contain a dictionary. Cannot extract weights.")
        return
    
    # If the model_data is a state_dict (has 'state_dict' key)
    if 'state_dict' in model_data:
        model_data = model_data['state_dict']
    
    # Find the final layer weights using the improved function
    final_layer_weights, layer_name = find_final_layer_weights(model_data)
    
    if final_layer_weights is None:
        print("Final layer weights not found in the model checkpoint.")
        print("Available keys:")
        for key in model_data.keys():
            print(f"  - {key}")
        return
    
    print(f"Using weights from layer: {layer_name}")
    
    # Check if the weights need to be transposed
    # If shape is (input_features, output_classes), transpose to (output_classes, input_features)
    if len(final_layer_weights.shape) == 2 and final_layer_weights.shape[0] > final_layer_weights.shape[1]:
        final_layer_weights = final_layer_weights.T
        print(f"Transposed weights to shape: {final_layer_weights.shape}")

    # Compute cosine similarity matrix
    cosine_sim_matrix = compute_cosine_similarity(final_layer_weights)

    # Create appropriate filenames based on epoch if applicable
    layer_suffix = layer_name.replace('.', '_').replace('/', '_')
    if epoch_num is not None:
        json_filename = f"cosine_similarity_{layer_suffix}_epoch_{epoch_num}.json"
        heatmap_filename = f"cosine_similarity_{layer_suffix}_epoch_{epoch_num}.png"
    else:
        json_filename = f"cosine_similarity_{layer_suffix}.json"
        heatmap_filename = f"cosine_similarity_{layer_suffix}.png"
    
    # Save similarity matrix as JSON
    similarity_json_path = model_folder / json_filename
    with open(similarity_json_path, "w") as f:
        json.dump(cosine_sim_matrix.tolist(), f, indent=4)
    
    print(f"Cosine similarity matrix saved to {similarity_json_path}")

    # Generate and save heatmap visualization
    heatmap_path = model_folder / heatmap_filename
    visualize_cosine_similarity(cosine_sim_matrix, heatmap_path, model_info, layer_name)
    print(f"Cosine similarity heatmap saved to {heatmap_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate cosine similarity of final-layer weights in a trained model.")
    
    # Support both positional and --folder argument styles
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("model_folder", type=str, nargs="?", help="Path to the saved model folder (positional argument)")
    group.add_argument("--folder", type=str, help="Path to the saved model folder (named argument)")
    
    args = parser.parse_args()
    
    # Use whichever argument is provided
    folder_path = args.folder if args.folder else args.model_folder
    
    evaluate_cosine_similarity(folder_path)