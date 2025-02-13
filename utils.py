import numpy as np
import torch
import random
from tqdm import tqdm  # For progress tracking
from config import config

# def set_seed(seed):
#     """Ensure reproducibility."""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def compute_prs(model, dataloader, device):
#     """Compute the Populated Region Set (PRS) specifically for the nn.Linear(512, 128) layer."""
#     set_seed(config["seed"])  # 🔹 Set seed here
#     model.eval()
#     decision_regions = set()
#     save_threshold = 1e6  # Threshold to save and clear patterns
#     save_file = "decision_regions_penultimate.txt"

#     # Clear the saved file before starting a new PRS computation
#     with open(save_file, "w") as f:
#         pass

#     with torch.no_grad():
#         for inputs, _ in tqdm(dataloader, desc="PRS Calculation for Penultimate Layer", leave=False):
#             inputs = inputs.to(device)

#             # Pass inputs through the entire classifier up to the penultimate layer
#             features = model.features(inputs)  # Extract features
#             features = model.classifier[0](features)  # Apply Flatten
#             features = model.classifier[1](features)  # Apply Linear(64*4*4 -> 512)
#             features = model.classifier[2](features)  # Apply ReLU
#             outputs = model.classifier[3](features)  # Apply Linear(512 -> 128)

#             # Compute binary activation patterns
#             signs = torch.sign(outputs)
#             signs[signs == 0] = -1  # Replace 0 with -1 for consistency

#             # Convert activations to tuples and add to decision regions
#             signs_cpu = signs.cpu().numpy()
#             for row in signs_cpu:
#                 decision_regions.add(tuple(row.tolist()))  # Ensure row is hashable

#             # Periodically save patterns to disk
#             if len(decision_regions) > save_threshold:
#                 with open(save_file, "a") as f:
#                     for pattern in decision_regions:
#                         f.write(f"{pattern}\n")
#                 decision_regions.clear()

#         # Save any remaining patterns
#         if decision_regions:
#             with open(save_file, "a") as f:
#                 for pattern in decision_regions:
#                     f.write(f"{pattern}\n")

#     # Count unique patterns saved to disk
#     unique_patterns = set()
#     with open(save_file, "r") as f:
#         for line in f:
#             unique_patterns.add(line.strip())

#     # Log the number of unique patterns
#     print(f"Total unique patterns: {len(unique_patterns)}")

#     return len(unique_patterns)

def evaluate(model, dataloader, device):
    """Evaluate the model's accuracy on the given dataloader."""
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    model.train()  # Restore training mode (ensure it's done outside if necessary)
    return 100 * float(correct) / total  # Avoid integer division


def compute_prs(activations):
    """
    Compute the Populated Region Set (PRS) for the stored activations of the penultimate layer.

    Args:
        activations (numpy.ndarray): Activation maps retrieved from the penultimate layer.
    
    Returns:
        float: PRS ratio (number of unique activation patterns divided by total samples).
    """
    decision_regions = set()

    # Compute binary activation patterns (sign-based)
    signs = np.sign(activations)
    signs[signs == 0] = +1  # Replace 0 with -1 for consistency

    # Convert each row into a tuple for uniqueness tracking
    for row in signs:
        decision_regions.add(tuple(row))

    # Compute PRS ratio as the number of unique patterns 
    prs_ratio = len(decision_regions)

    return prs_ratio


def compute_unique_activations(activations):
    """
    Compute the number of unique activation patterns for the stored activations of the penultimate layer.

    Args:
        activations (numpy.ndarray): Activation maps retrieved from the penultimate layer.
    
    Returns:
        int: Number of unique activation patterns.
    """
    if activations is None or len(activations) == 0:
        print("⚠️ Warning: No activations received in compute_unique_activations. Returning 0.")
        return 0

    print(f"🔹 Computing unique activation patterns for {activations.shape[0]} samples...")

    decision_regions = set()
    

    # Compute binary activation patterns (sign-based)
    signs = np.sign(activations)
    signs[signs == 0] = -1  

    # Debugging: Show the first few activation patterns
    print(f"🔹 First 5 activation rows (before binarization):\n{activations[:5]}")
    print(f"🔹 First 5 binarized patterns:\n{signs[:5]}")

    # Convert each row into a tuple for uniqueness tracking
    for row in signs:
        decision_regions.add(tuple(row))

    unique_count = len(decision_regions)
    print(f"🔹 Total Unique Activation Patterns Found: {unique_count}")

    return unique_count

