config = {
    "datasets": ["CIFAR10", "F-MNIST", "MNIST"],  # Iterate through multiple datasets
    "batch_sizes": [64, 128, 512, 2048],  # Batch sizes to iterate over
    "epochs": 300,  # Number of epochs for training
    "learning_rate": 1e-3,  # Learning rate for the optimizer
    "model_save_path": "models/",  # Directory to save trained models
    "results_save_path": "results/",  # Directory to save training results
    "device": "cuda:0",  # Change to 'cpu' if GPU is unavailable
    "seed": 375,  # Seed for reproducibility
    "test_batch_size": 256, 
}

# "CIFAR10", "F-MNIST"