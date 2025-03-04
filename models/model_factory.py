import os
import torch
import torch.nn as nn
import torchvision.models as models
from models import CustomCNN  # Your existing CNN-6 model

MODEL_SAVE_DIR = "models/saved"

def get_model(model_name, input_channels=3):
    """
    Returns the appropriate model based on the provided model_name.

    Args:
        model_name (str): The name of the model. Options: ['CNN-6', 'VGG16', 'ResNet18']
        input_channels (int): Number of input channels (1 for MNIST/F-MNIST, 3 for CIFAR-10)

    Returns:
        torch.nn.Module: The selected model with minimal modifications.
    """
    if model_name == "CNN-6":
        return CustomCNN(input_channels)

    elif model_name == "VGG16":
        model = models.vgg16(weights=None)  # No pre-trained weights
        if input_channels != 3:
            model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)

        # Reduced fc1/fc2 but kept the structure
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),  # Reduced from 4096 → 2048
            nn.ReLU(),
            nn.Linear(2048, 1024),  # Reduced from 4096 → 1024
            nn.ReLU(),
            nn.Linear(1024, 10)  # Output remains the same
        )

        return model

    elif model_name == "ResNet18":
        model = models.resnet18(weights=None)  # No pre-trained weights
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Replace only the final fc layer (1000 → 10 classes)
        model.fc = nn.Linear(model.fc.in_features, 10)

        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from ['CNN-6', 'VGG16', 'ResNet18']")

def load_trained_model(model_name, dataset_name, batch_size, device="cuda", prs_enabled=False, warmup_epochs=50):
    """
    Loads a trained model from the saved directory.

    Args:
        model_name (str): Model architecture name (e.g., "CNN-6", "VGG16", "ResNet18").
        dataset_name (str): Dataset name (e.g., "CIFAR10").
        batch_size (int): Batch size used during training.
        device (str): Device to load model onto ("cuda" or "cpu").

    Returns:
        torch.nn.Module: The trained model with loaded weights.
    """
    if prs_enabled:
        model_path = os.path.join("models", "saved", f"{model_name}_{dataset_name}_batch_{batch_size}_warmup_{warmup_epochs}_PRS", "model.pth")
    else:
        model_path = os.path.join("models", "saved", f"{model_name}_{dataset_name}_batch_{batch_size}", "model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Initialize model architecture
    model = get_model(model_name, input_channels=3)

    # Load trained weights only
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"Loaded trained model from {model_path}")
    return model
