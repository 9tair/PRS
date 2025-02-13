import torch
import torch.optim as optim
import torch.nn as nn  
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  
import json
import numpy as np
import random
import os

from models.cnn import CustomCNN
from utils import compute_unique_activations, evaluate
from config import config

# Function to set the seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_datasets(dataset_name):
    """Return dataset and transformations based on dataset_name"""
    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
        ])
        train_dataset = datasets.CIFAR10(root="data/", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="data/", train=False, download=True, transform=transform)
        input_channels = 3
    elif dataset_name in ["MNIST", "F-MNIST"]:
        transform = transforms.Compose([
            transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,) if dataset_name == "MNIST" else (0.2860,),
                std=(0.3081,) if dataset_name == "MNIST" else (0.3530,)
            )
        ])

        dataset_class = datasets.MNIST if dataset_name == "MNIST" else datasets.FashionMNIST
        train_dataset = dataset_class(root="data/", train=True, download=True, transform=transform)
        test_dataset = dataset_class(root="data/", train=False, download=True, transform=transform)
        input_channels = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset, input_channels

def train():
    set_seed(config["seed"])

    results = {}

    for batch_size in tqdm(config["batch_sizes"], desc="Batch Sizes Loop"):
        for dataset_name in tqdm(config["datasets"], desc=f"Training on Different Datasets"):
            train_dataset, test_dataset, input_channels = get_datasets(dataset_name)

            generator = torch.Generator()
            generator.manual_seed(config["seed"])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(config["seed"]), generator=generator)
            test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

            model = CustomCNN(input_channels).to(config["device"])
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0) #1e-4
            criterion = nn.CrossEntropyLoss()

            scaler = torch.amp.GradScaler("cuda")

            metrics = {
                "epoch": [],
                "train_accuracy": [],
                "test_accuracy": [],
                "prs_ratios": []
            }

            epoch_activations = {}

            activations = {"penultimate": []}
            def hook(module, input, output):
                activations["penultimate"].append(output.detach().cpu().numpy())

            hook_handle = model.classifier[3].register_forward_hook(hook)

            for epoch in tqdm(range(config["epochs"]), desc=f"Training {dataset_name} | Batch {batch_size}"):
                model.train()
                epoch_loss = 0
                correct_train = 0
                total_train = 0
                activations["penultimate"].clear()

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

                    optimizer.zero_grad()

                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()

                    _, predicted = outputs.max(1)
                    correct_train += (predicted == labels).sum().item()
                    total_train += labels.size(0)

                epoch_activations[f"epoch_{epoch+1}"] = np.concatenate(activations["penultimate"], axis=0)

                prs_ratio = compute_unique_activations(epoch_activations[f"epoch_{epoch+1}"]) / len(train_dataset)
                metrics["prs_ratios"].append(prs_ratio)
                tqdm.write(f"Epoch {epoch+1}: PRS Ratio = {prs_ratio:.4f}")

                test_accuracy = evaluate(model, test_loader, config["device"])

                train_accuracy = 100 * correct_train / total_train

                metrics["epoch"].append(epoch + 1)
                metrics["train_accuracy"].append(train_accuracy)
                metrics["test_accuracy"].append(test_accuracy)

                tqdm.write(f"Epoch {epoch+1}/{config['epochs']} - Loss: {epoch_loss/len(train_loader):.4f} - Train Acc: {train_accuracy:.2f}% - Test Acc: {test_accuracy:.2f}%")

            hook_handle.remove()

            os.makedirs(config["results_save_path"], exist_ok=True)
            np.save(f"{config['results_save_path']}activations_{dataset_name}_batch_{batch_size}.npy", epoch_activations)

            tqdm.write(f"Completed Training {dataset_name} | Batch {batch_size}: Saving Metrics")
            results[f"{dataset_name}_batch_{batch_size}"] = metrics
            metrics_path = f"{config['results_save_path']}metrics_{dataset_name}_batch_{batch_size}.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

if __name__ == "__main__":
    train()
