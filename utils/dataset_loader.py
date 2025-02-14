import torch
from torchvision import datasets, transforms

def get_datasets(dataset_name):
    """Return dataset and transformations based on dataset_name"""
    dataset_paths = "datasets/"

    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
        ])
        train_dataset = datasets.CIFAR10(root=dataset_paths, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=dataset_paths, train=False, download=True, transform=transform)
        input_channels = 3
    elif dataset_name in ["MNIST", "F-MNIST"]:
        transform = transforms.Compose([
            transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,) if dataset_name == "MNIST" else (0.2860,), 
                                    std=(0.3081,) if dataset_name == "MNIST" else (0.3530,))
        ])
        dataset_class = datasets.MNIST if dataset_name == "MNIST" else datasets.FashionMNIST
        train_dataset = dataset_class(root=dataset_paths, train=True, download=True, transform=transform)
        test_dataset = dataset_class(root=dataset_paths, train=False, download=True, transform=transform)
        input_channels = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset, input_channels
