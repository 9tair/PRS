import torch

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
