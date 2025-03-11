import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from pathlib import Path

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

def visualize_cosine_similarity(similarity_matrix, save_path):
    """
    Create and save a heatmap visualization of the cosine similarity matrix.

    Args:
        similarity_matrix (numpy.ndarray): Cosine similarity matrix.
        save_path (str): Path to save the heatmap image.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Cosine Similarity of Final Layer Weights")
    plt.xlabel("Class Index")
    plt.ylabel("Class Index")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_cosine_similarity(model_folder):
    """
    Evaluate cosine similarity of the final classification layer in the model.

    Args:
        model_folder (str): Path to the model's saved folder.
    """
    model_folder = Path(model_folder)
    
    # Look for the model file (assuming it's a .pth or .pt file)
    model_files = list(model_folder.glob("*.pth")) + list(model_folder.glob("*.pt"))
    if not model_files:
        print(f"No model checkpoint found in {model_folder}")
        return
    
    model_path = model_files[0]  # Use the first found model file
    model_data = torch.load(model_path, map_location=torch.device("cpu"))
    
    # Extract the final layer weights (assuming standard naming conventions)
    final_layer_weights = None
    for key, value in model_data.items():
        if "fc.weight" in key or "classifier.weight" in key or "linear.weight" in key:
            final_layer_weights = value.cpu().numpy()
            break
    
    if final_layer_weights is None:
        print("Final layer weights not found in the model checkpoint.")
        return

    # Compute cosine similarity matrix
    cosine_sim_matrix = compute_cosine_similarity(final_layer_weights)

    # Save similarity matrix as JSON
    similarity_json_path = model_folder / "cosine_similarity.json"
    with open(similarity_json_path, "w") as f:
        json.dump(cosine_sim_matrix.tolist(), f, indent=4)
    
    print(f"Cosine similarity matrix saved to {similarity_json_path}")

    # Generate and save heatmap visualization
    heatmap_path = model_folder / "cosine_similarity.png"
    visualize_cosine_similarity(cosine_sim_matrix, heatmap_path)
    print(f"Cosine similarity heatmap saved to {heatmap_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate cosine similarity of final-layer weights in a trained model.")
    parser.add_argument("model_folder", type=str, help="Path to the saved model folder (e.g., models/saved/CNN-6_CIFAR10_batch_128)")
    
    args = parser.parse_args()
    evaluate_cosine_similarity(args.model_folder)
