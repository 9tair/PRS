
import torch
import torch.optim as optim
import torch.nn as nn  # Added the missing import
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # For progress tracking
import json

from models.cnn import CustomCNN
from utils import compute_prs, evaluate
from config import config

def train():
    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),  # Bilinear upsampling
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root="data/", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="data/", train=False, download=True, transform=transform)

    results = {}

    for batch_size in config["batch_sizes"]:
        # Data Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        # Model, Optimizer, and Loss Function
        model = CustomCNN().to(config["device"])
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])
        criterion = nn.CrossEntropyLoss()

        # Data to store metrics for visualization
        metrics = {
            "epoch": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "prs_ratio": []
        }

        # Training Loop
        for epoch in tqdm(range(config["epochs"]), desc=f"Training with batch size {batch_size}"):
            model.train()
            epoch_loss = 0
            correct_train = 0
            total_train = 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False):
                inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Train accuracy calculation
                _, predicted = outputs.max(1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            scheduler.step()

            # Compute PRS Ratio
            prs_ratio = compute_prs(model, train_loader, config["device"]) / len(train_dataset)

            # Compute Test Accuracy
            test_accuracy = evaluate(model, test_loader, config["device"])

            # Compute Train Accuracy
            train_accuracy = 100 * correct_train / total_train

            # Log metrics
            metrics["epoch"].append(epoch + 1)
            metrics["train_accuracy"].append(train_accuracy)
            metrics["test_accuracy"].append(test_accuracy)
            metrics["prs_ratio"].append(prs_ratio)

            tqdm.write(f"Epoch {epoch+1}/{config['epochs']} - Loss: {epoch_loss/len(train_loader):.4f} - Train Acc: {train_accuracy:.2f}% - Test Acc: {test_accuracy:.2f}% - PRS: {prs_ratio:.4f}")

        # Save metrics for this batch size
        results[f"CIFAR10_batch_{batch_size}"] = metrics

    # Save Results
    with open(config["results_save_path"] + "metrics.json", "w") as f:
        json.dump(results, f)
