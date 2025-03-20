# config = {
#     "models": ["CNN-6", "VGG16", "ResNet18"],  # Iterate through all model architectures
#     "datasets": ["CIFAR10", "F-MNIST", "MNIST"],  # Iterate through all datasets
#     "batch_sizes": [64, 128, 512, 1024, 2048],  # Iterate through multiple batch sizes
#     "epochs": 300,  # Set to the required number of epochs
#     "learning_rate": 1e-3,  # Learning rate for the optimizer
#     "model_save_path": "models/",  # Directory to save trained models
#     "results_save_path": "results/",  # Directory to save training results
#     "device": "cuda:3",  # Change to 'cpu' if GPU is unavailable
#     "seed": 375,  # Seed for reproducibility
#     "test_batch_size": 256,  # Batch size for evaluation
# }

# "F-MNIST", "MNIST"


# config = {
#     "datasets": ["MNIST"],  # Iterate through multiple datasets
#     "batch_sizes": [64, 128, 512, 2048],  # Batch sizes to iterate over
#     "epochs": 300,  # Number of epochs for training
#     "learning_rate": 1e-3,  # Learning rate for the optimizer
#     "model_save_path": "models/",  # Directory to save trained models
#     "results_save_path": "results/",  # Directory to save training results
#     "device": "cuda:3",  # Change to 'cpu' if GPU is unavailable
#     "seed": 375,  # Seed for reproducibility
#     "test_batch_size": 256, 
# }


config = {
    "models": ["VGG16"],  # Iterate through all model architectures
    "datasets": ["CIFAR10"],  # Iterate through all datasets
    "batch_sizes": [128], # Iterate through multiple batch sizes
    "epochs": 300,  # Set to the required number of epochs
    "warmup_epochs": 50,
    "learning_rate": 1e-3,  # Learning rate for the optimizer
    "model_save_path": "models/",  # Directory to save trained models
    "results_save_path": "results/",  # Directory to save training results
    "device": "cuda:3",  # Change to 'cpu' if GPU is unavailable
    "seed": 375,  # Seed for reproducibility
    "test_batch_size": 256,  # Batch size for evaluation
    "lambda_std": 0.2,
    "lambda_mrv": 0.8,
    "lambda_hamming": 1,
    "rdr_agreement_threshold": 0.1,
}