import torch.nn as nn

def initialize_weights(model):
    """Apply improved weight initialization to model."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming initialization for convolutional layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # Standard initialization for batch normalization
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Xavier initialization for fully connected layers
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model
