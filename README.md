## 🏗️ Folder Structure

.
├── config.py # Central configuration for training parameters
├── train_cnn.py # Main script for training CNN-6 models (baseline)
├── train_prs_topk_models.py # Train and save top-K models by PRS ratio
├── train_vgg16_resnet18.py # Main script for training VGG16/ResNet18 (baseline)
├── train_with_pals_regularization.py # Train with PaLS regularization
├── train_with_prs_regularization_cnn.py # Train CNN-6 with PRS regularizer
├── train_vgg16_prs_regularization.py # Train VGG16/ResNet18 with PRS regularizer (needs to be distinct or merged)
├── models/
│ ├── init.py
│ └── model_factory.py # Functions to create model architectures (CNN-6, VGG16, ResNet18)
│ └── saved/ # Default directory for saved models, metrics, and region data
│ └── <MODEL><DATASET>batch<BSIZE><TAGS>/
│ ├── config.json
│ ├── metrics.json
│ └── epoch_<EPOCH_NUM>/
│ ├── model.pth
│ ├── optimizer.pth
│ ├── scheduler.pth
│ ├── major_regions.json
│ └── unique_patterns.json
├── utils/
│ ├── init.py
│ ├── adversarial_attacks.py # Implementations of FGSM, PGD, BIM, CW, AutoAttack
│ ├── compute_unique_activations.py # Computes unique binary activation patterns (for PRS)
│ ├── evaluate.py # (Assumed) Standard evaluation function
│ ├── freeze_final_layer.py # Utility to freeze the final classification layer
│ ├── get_datasets.py # (Assumed) Functions to load and preprocess datasets
│ ├── hooks.py # Registers forward hooks to capture activations
│ ├── initialize_weights.py # (Assumed) Model weight initialization utility
│ ├── logger.py # Setup for logging
│ ├── mr_tracker.py # Computes Major Regions, MRVs, etc.
│ ├── regularization.py # Implements L_MRV, L_ham, PaLS loss functions
│ └── save_model_utils.py # Utilities for saving checkpoints and metadata
├── evaluation/
│ ├── adv_evaluate_robustness.py # Script to evaluate adversarial robustness of trained models
│ ├── visualize_mr_er.py # Script to visualize Major/Extra Region distributions
│ └── visualize_per_file.py # (Appears to be a duplicate of adv_evaluate_robustness.py)
└── README.md # This file


---

## ⚙️ Configuration

All training and evaluation settings are controlled via `config.py`, including:

- Model type, dataset, batch size
- Number of epochs, learning rates
- Regularization weights (`lambda_mrv`, `lambda_hamming`, `pals_lambda`)
- Warm-up settings and paths
- Device and seed control

---

## 🚀 Usage

### 1. Train Baseline Models
```bash
# CNN-6
python train_cnn.py

# VGG-16 / ResNet-18
python train_vgg16_resnet18.py


