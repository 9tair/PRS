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
        torch.nn.Module: The selected model with a modified classifier (not pre-trained)
    """
    if model_name == "CNN-6":
        return CustomCNN(input_channels)

    elif model_name == "VGG16":
        model = models.vgg16(weights=None)  
        model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)  # Adjust input channels
        model.classifier = nn.Sequential(
            nn.Linear(25088, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Adjust output classes
        )
        return model

    elif model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjust input channels
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Adjust output classes
        )
        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from ['CNN-6', 'VGG16', 'ResNet18']")
