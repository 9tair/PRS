config = {
    # --- Core Model and Data Settings ---
    "models": ["CNN-6"],  # List of model architectures to train (e.g., "CNN-6", "VGG16", "ResNet18")
    "datasets": ["CIFAR10"],  # List of datasets to use (e.g., "CIFAR10", "F-MNIST", "MNIST")
    "batch_sizes": [128],  # List of batch sizes for training iterations

    # --- Training Duration and Phases ---
    "epochs": 300,  # Total number of training epochs for the main (PRS/PaLS) phase
    "warmup_epochs": 400, # Number of epochs for the initial warm-up phase (standard training)
                          # If a checkpoint from this epoch exists, warm-up is skipped.

    # --- Optimizer and Learning Rate Settings ---
    "learning_rate": 1e-3,  # Learning rate for the optimizer during the warm-up phase
    "learning_rate_prs": 0.0001, # Learning rate for the optimizer during the PRS/PaLS regularization phase
                                 # (after warm-up and potential final layer freeze)

    # --- Paths and Device ---
    "model_save_path": "models/",  # Base directory to save trained model checkpoints and associated data
    "results_save_path": "results/",  # Base directory to save overall training metrics and results (if applicable)
    "device": "cuda:0",  # Computation device (e.g., "cuda:0", "cuda:1", "cpu")

    # --- Reproducibility and Evaluation ---
    "seed": 375,  # Random seed for reproducibility across runs
    "test_batch_size": 256,  # Batch size for evaluating the model on the test set

    # --- Regularization Hyperparameters (Loss Components) ---
    "label_smoothing": 0.2, # Amount of label smoothing for the Cross-Entropy loss (0.0 for no smoothing)

    "lambda_ce": 0.2,       # Weight for the standard Cross-Entropy loss component (during PRS/PaLS phase)
                            # The paper (Fig 1c) shows CE loss + PRS Reg.
                            # This implies lambda_ce for the PRS phase might be different from implicit 1.0 during warmup.

    # PRS Regularizer Specific (L_MRV and L_ham from the paper)
    "lambda_mrv": 0,        # Weight for the MRV loss (L_MRV in the paper, Eq. 1)
                            # Encourages features to be close to their class's Major Region Mean Vector.
    "lambda_hamming": 0,    # Weight for the Hamming distance loss (L_ham in the paper, Eq. 2)
                            # Encourages feature sign patterns to align with MRV-derived patterns.

    # PaLS Regularizer Specific (Prototype-aligned Logit Squeezing)
    "use_pals": True,       # Boolean flag to enable/disable PaLS regularization
    "pals_lambda": 0.8,     # Weight for the PaLS loss component (λ_PaLS or similar in PaLS-related works)
                            # Encourages penultimate layer features to align with class prototypes (MRVs).

    # --- PRS/MR Computation Specific ---
    "rdr_agreement_threshold": 0.7, # Threshold for sign consistency when computing Relaxed Decision Region (RDR) masks
                                    # (Mentioned in your `mr_tracker.py` but not directly in the paper's PRS regularizer)
                                    # This might be for a more advanced/experimental version of MRV/RRV computation.

    # --- Mixed Precision Training ---
    "use_amp": False,  # Boolean flag to enable/disable Automatic Mixed Precision (AMP) training for speed-up on compatible GPUs.

    # --- Potentially Missing or Implied Configs (Consider adding if used elsewhere) ---
    # "lambda_std": 0.2, # This was present in your input but its usage is unclear from the filenames.
                         # If it's a standard deviation for some noise or a different regularizer,
                         # it needs to be documented or integrated. Assuming it's not directly used by core PRS/PaLS for now.

    # "recompute_mr": True, # (Example) Flag to control if Major Regions are recomputed every epoch during PRS phase.
                           # Your `train_with_pals_regularization.py` has logic for this.

    # "top_k_prs_save": 5, # (Example, from `train_prs_topk_models.py`) Number of top models to save based on PRS ratio.

    # "lr_step_size": 10, # (Example, from `train_with_pals_regularization.py`) For StepLR scheduler
    # "lr_gamma": 0.1,    # (Example, from `train_with_pals_regularization.py`) For StepLR scheduler
    # "lr_step_size_prs": 10, # (Example) For StepLR scheduler during PRS phase
    # "lr_gamma_prs": 0.1,    # (Example) For StepLR scheduler during PRS phase

    # "save_interval": 50, # (Example) Frequency of saving checkpoints during training.
}