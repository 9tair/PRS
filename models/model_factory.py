import torch.nn as nn
import torchvision.models as models
from models.cnn import CustomCNN  # Your existing CNN-6 model

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

        # Keep fc1 and fc2, only replace the final fc3 layer (1000 → 10 classes)
        model.classifier[6] = nn.Linear(4096, 10)  # Minimal modification

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
