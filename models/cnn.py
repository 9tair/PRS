import torch.nn as nn
import torch

class CustomCNN(nn.Module):
    def __init__(self, input_channels=3):  # Default to RGB (CIFAR-10)
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),  # ReLU BEFORE pooling
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Pooling AFTER ReLU

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Pooling AFTER ReLU

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_layer_activations(self, x, layer_idx):
        """Extract activations from a specific meaningful layer."""
        meaningful_layers = []

        # Collect meaningful layers (Conv2d and Linear)
        for layer in self.features:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                meaningful_layers.append(layer)
        for layer in self.classifier[:-1]:  # Exclude the last layer
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                meaningful_layers.append(layer)

        # Pass input through layers until the specified layer
        for idx, layer in enumerate(meaningful_layers):
            x = layer(x)
            if idx == layer_idx:
                # Flatten if output is multi-dimensional
                if x.ndim > 2:
                    x = x.view(x.size(0), -1)
                return x

        raise ValueError(f"Invalid layer index: {layer_idx}")

class ActivationExtractor(nn.Module):
    """ Wrapper to extract activations from a specific layer BEFORE ReLU. """
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        
        # Hook to capture pre-ReLU activation
        def hook(module, input, output):
            self.activation = input[0].detach().clone()

        # Attach the hook to the correct layer
        self.hook = self.model.features[target_layer].register_forward_hook(hook)

    def forward(self, x):
        _ = self.model(x)  # Forward pass to trigger hook
        return self.activation

