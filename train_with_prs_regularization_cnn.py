import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────
#  project‑specific utilities
# ──────────────────────────────────────────────────────────────────────────
from models import get_model
from utils import (
    get_datasets, evaluate, compute_unique_activations,
    register_activation_hook, compute_major_regions,
    save_model_checkpoint, set_seed, freeze_final_layer
)
from utils.logger import setup_logger
from utils.regularization import compute_mrv_loss, compute_hamming_loss
from config import config

# ──────────────────────────────────────────────────────────────────────────
#  warm‑up checkpoint loader
# ──────────────────────────────────────────────────────────────────────────
def load_checkpoint_if_exists(model, optimizer, modelname,
                              dataset_name, batch_size, logger):
    # Signature: Returns 6 items
    logger.debug(f"[Checkpoint Loader] Attempting to load for {modelname}, {dataset_name}, batch {batch_size}.")
    ckpt_dir = os.path.join("models", "saved", f"{modelname}_{dataset_name}_batch_{batch_size}_top_prs_rank1")
    warm_ep = config.get("warmup_epochs", 0)
    ep_dir = os.path.join(ckpt_dir, f"epoch_{warm_ep}")

    if not os.path.exists(ep_dir):
        logger.warning(f"[Checkpoint Loader] Warm‑up checkpoint directory not found: {ep_dir} – starting from scratch.")
        return False, model, optimizer, None, None, None # 6 items

    logger.info(f"[Checkpoint Loader] Loading warm‑up checkpoint from: {ep_dir}")
    try:
        model_path = os.path.join(ep_dir, "model.pth")
        optimizer_path = os.path.join(ep_dir, "optimizer.pth")
        scheduler_path = os.path.join(ep_dir, "scheduler.pth")
        major_regions_path = os.path.join(ep_dir, "major_regions.json")
        unique_patterns_path = os.path.join(ep_dir, "unique_patterns.json")

        if not os.path.exists(model_path):
            logger.error(f"[Checkpoint Loader] Model file missing: {model_path}")
            return False, model, optimizer, None, None, None
        model.load_state_dict(torch.load(model_path, map_location=config.get("device", "cpu")))
        
        if not os.path.exists(optimizer_path):
            logger.error(f"[Checkpoint Loader] Optimizer file missing: {optimizer_path}")
            return False, model, optimizer, None, None, None
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=config.get("device", "cpu")))
        logger.debug(f"[Checkpoint Debug] Model state dict keys (sample): {list(model.state_dict().keys())[:5]}")
        logger.debug(f"[Checkpoint Debug] Optimizer param group count: {len(optimizer.param_groups)}")


        sched_state = None
        if os.path.exists(scheduler_path):
            sched_state = torch.load(scheduler_path, map_location=config.get("device", "cpu"))
        else:
            logger.warning(f"[Checkpoint Loader] Scheduler file {scheduler_path} not found.")
            
        if not os.path.exists(major_regions_path):
            logger.error(f"[Checkpoint Loader] Major regions file missing: {major_regions_path}")
            return False, model, optimizer, None, None, None 
        with open(major_regions_path) as f:
            major_regions = json.load(f)
            
        sanitize_major_regions(major_regions, logger)

        unique_patterns = None
        if os.path.exists(unique_patterns_path):
            with open(unique_patterns_path) as f:
                unique_patterns = json.load(f)
        else:
            logger.info(f"[Checkpoint Loader] Unique patterns file {unique_patterns_path} not found.")

        model.to(config.get("device", "cpu")).train()
        logger.info("[Checkpoint Loader] Checkpoint loaded successfully.")
        return True, model, optimizer, sched_state, major_regions, unique_patterns # 6 items
    except Exception as e:
        logger.error(f"[Checkpoint Loader] Checkpoint load failed with exception: {e}", exc_info=True)
        return False, model, optimizer, None, None, None

def sanitize_major_regions(major_regions, logger, max_norm=100.0):
    """
    Clip and sanitize MRVs to prevent unstable training during PRS phase.
    """
    for cls_key, entry in major_regions.items():
        mrv = entry.get("mrv", None)
        if mrv is None:
            logger.warning(f"[MRV Sanity] Missing MRV for {cls_key}.")
            continue

        mrv = np.array(mrv)

        if not np.isfinite(mrv).all():
            logger.warning(f"[MRV Sanity] MRV for {cls_key} has NaN/Inf. Zeroing.")
            mrv = np.zeros_like(mrv)

        norm = np.linalg.norm(mrv)
        if norm > max_norm:
            logger.warning(f"[MRV Sanity] MRV for {cls_key} norm={norm:.2f} too large. Clipping.")
            mrv = mrv / norm * max_norm

        major_regions[cls_key]["mrv"] = mrv

# ──────────────────────────────────────────────────────────────────────────
#  main routine
# ──────────────────────────────────────────────────────────────────────────
def train():
    set_seed(config.get("seed", 42))
    results = {}

    for modelname in tqdm(config.get("models", []), desc="Model"):
      for dataset_name in tqdm(config.get("datasets", []), desc="Dataset"):
       for batch_size   in tqdm(config.get("batch_sizes", []), desc="Batch size"):

        logger = setup_logger(modelname, dataset_name, batch_size)
        logger.info(f"===== Starting Training Run: Model={modelname}, Dataset={dataset_name}, BatchSize={batch_size} =====")
        logger.info(f"[Config] Seed: {config.get('seed', 'Not Set')}, Device: {config.get('device', 'cpu')}")
        logger.info(f"[Config] Epochs: Warmup={config.get('warmup_epochs', 0)}, Total={config.get('epochs', 0)}")
        logger.info(f"[Config] Learning Rate: {config.get('learning_rate', 0.001)}")
        logger.info(f"[Config] Lambdas: CE={config.get('lambda_ce', 1.0)}, MRV={config.get('lambda_mrv', 0.0)}, Hamming={config.get('lambda_hamming', 0.0)}")
        if config.get('lambda_hamming', 0.0) > 0:
             logger.info(f"[Config] Hamming Margin (if applicable): {config.get('hamming_margin', 0.1)}")

        train_ds, test_ds, in_ch = get_datasets(dataset_name)
        num_classes_val = 10 # Default
        if hasattr(train_ds, 'classes') and train_ds.classes is not None:
            num_classes_val = len(train_ds.classes)
        elif hasattr(train_ds, 'dataset') and hasattr(train_ds.dataset, 'classes') and train_ds.dataset.classes is not None:
            num_classes_val = len(train_ds.dataset.classes)
        num_classes = num_classes_val

        logger.info(f"[Data Setup] Training samples: {len(train_ds)}, Test samples: {len(test_ds)}, Input channels: {in_ch}, Num classes: {num_classes}")
        sample_input, sample_label = train_ds[0]
        logger.debug(f"[Dataset Sample] Input shape: {sample_input.shape}, Label: {sample_label}")

        
        train_loader = DataLoader(
            train_ds, batch_size, shuffle=True,
            worker_init_fn=lambda worker_id: np.random.seed(config.get("seed", 42) + worker_id),
            generator=torch.Generator().manual_seed(config.get("seed", 42))
        )
        test_loader  = DataLoader(test_ds, config.get("test_batch_size", batch_size), shuffle=False)
        
        model = get_model(modelname, in_ch, num_classes=num_classes).to(config.get("device", "cpu"))
        
        for n, p in model.named_parameters():
            if not torch.isfinite(p).all():
                logger.error(f"[Sanity] parameter {n} contains NaN / Inf!")
                
        use_amp = False                           # disable autocast
        scaler  = torch.amp.GradScaler(enabled=False)   # disable GradScaler
        logger.info("[Patch] AMP disabled – running in pure FP32 for this trial")

        logger.debug(f"[Model Summary] Architecture:\n{model}")

        
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
        criterion = nn.CrossEntropyLoss()
        use_amp = config.get("use_amp", torch.cuda.is_available())
        scaler = torch.amp.GradScaler(enabled=use_amp)
        
        scheduler_step_size = config.get("lr_step_size", 10)
        scheduler_gamma = config.get("lr_gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        activations = {"penultimate": [], "skip_batch": False, "current": None}
        hook_handle = register_activation_hook(model, activations, modelname, dataset_name, batch_size, logger)
        if hook_handle is None:
            logger.error("[Hook Setup] Failed to register activation hook! Skipping this run.")
            continue 

        warmup_ep = config.get("warmup_epochs", 0)
        total_ep = config.get("epochs", 0)
        save_interval = config.get("save_interval", 1)
        save_ep = {*range(warmup_ep, total_ep + 1, save_interval), total_ep}
        if warmup_ep > 0 and warmup_ep not in save_ep : save_ep.add(warmup_ep)

        major_regions = None
        # Corrected: load_checkpoint_if_exists returns 6 items
        resumed, model, optimizer, loaded_sched_state, major_regions_loaded, unique_patterns_loaded = load_checkpoint_if_exists(
            model, optimizer, modelname, dataset_name, batch_size, logger
        )
        
        max_abs_param = max(p.abs().max().item() for p in model.parameters())
        logger.info(f"[Sanity] Largest |param| in checkpoint = {max_abs_param:.1f}")

        if resumed:
            major_regions = major_regions_loaded
            logger.info("[DEBUG] Checking MRV values...")
            for cls, data in major_regions.items():
                mrv = np.array(data['mrv'])
                if not np.all(np.isfinite(mrv)):
                    logger.error(f"[DEBUG] MRV for class {cls} contains non-finite values: {mrv}")
                elif np.linalg.norm(mrv) > 1e4:
                    logger.error(f"[DEBUG] MRV for class {cls} norm too large: {np.linalg.norm(mrv):.2f}")

            logger.info("[Resuming] Successfully resumed from checkpoint.")
            if loaded_sched_state:
                scheduler.load_state_dict(loaded_sched_state)
                logger.info("[Resuming] Scheduler state loaded.")
        else:
            logger.info("[Resuming] No checkpoint found or failed to load, starting fresh.")

        # Stores list of label numpy arrays from the most recently completed training epoch.
        labels_from_last_full_epoch_for_mrv = [] 
        # Stores list of activation tensors from the most recently completed training epoch.
        activations_from_last_full_epoch_for_mrv = []


        if not resumed:
            logger.info("===== WARM‑UP PHASE =====")
            for epoch in range(warmup_ep):
                epoch_num_display = epoch + 1
                logger.info(f"--- Warm-up Epoch {epoch_num_display}/{warmup_ep} ---")
                model.train()
                
                current_epoch_batch_activations = [] # Collect activations for THIS epoch
                current_epoch_batch_labels = []      # Collect labels for THIS epoch

                activations["skip_batch"] = False # Ensure hook is active
                epoch_ce_loss = 0.0
                correct = 0
                total_samples_processed = 0

                for batch_idx, (inp, lbl) in enumerate(train_loader):
                    inp, lbl = inp.to(config.get("device", "cpu")), lbl.to(config.get("device", "cpu"))
                    optimizer.zero_grad(set_to_none=True)
                    activations["current"] = None 
                    if activations["current"] is not None:
                        logger.debug(f"[Activations] Hook captured tensor with shape: {activations['current'].shape}")
                    else:
                        logger.warning("[Activations] Hook failed to capture tensor. Check model/hook.")


                    with torch.amp.autocast(device_type=config.get("device", "cpu").split(':')[0], enabled=use_amp):
                        out  = model(inp)
                        ce_loss = criterion(out, lbl)
                    
                    if not torch.isfinite(ce_loss):
                        logger.error(f"Warm-up E{epoch_num_display} B{batch_idx+1} CE loss NaN/Inf. Skipping backward.")
                        continue

                    scaler.scale(ce_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_ce_loss += ce_loss.item() * inp.size(0)
                    correct += (out.argmax(1) == lbl).sum().item()
                    total_samples_processed += lbl.size(0)
                    
                    current_epoch_batch_labels.append(lbl.cpu().numpy())
                    if activations["current"] is not None:
                         current_epoch_batch_activations.append(activations["current"].detach().cpu())
                    else: # This case should be rare if hook is robust
                        logger.warning(f"Warm-up E{epoch_num_display} B{batch_idx+1} Hook did not set activations['current']!")
                
                # After epoch, store this epoch's data for potential MRV computation
                labels_from_last_full_epoch_for_mrv = current_epoch_batch_labels
                activations_from_last_full_epoch_for_mrv = current_epoch_batch_activations

                avg_epoch_ce_loss = epoch_ce_loss / total_samples_processed if total_samples_processed > 0 else 0
                train_acc = 100 * correct / total_samples_processed if total_samples_processed > 0 else 0
                
                current_epoch_prs = 0.0
                if activations_from_last_full_epoch_for_mrv: # Use the collected activations
                    try:
                        if not all(isinstance(t, torch.Tensor) for t in activations_from_last_full_epoch_for_mrv):
                            logger.error(f"Warm-up E{epoch_num_display} Non-tensor in acts_from_last_epoch")
                        elif not activations_from_last_full_epoch_for_mrv:
                             logger.warning(f"Warm-up E{epoch_num_display} acts_from_last_epoch is empty for PRS.")
                        else:
                            acts_np_epoch = torch.cat(activations_from_last_full_epoch_for_mrv).cpu().numpy()
                            if len(acts_np_epoch) > 0 :
                                unique_acts_count = compute_unique_activations(acts_np_epoch, logger)
                                current_epoch_prs = unique_acts_count / len(train_ds) if len(train_ds) > 0 else 0.0
                    except Exception as e:
                        logger.error(f"Warm-up E{epoch_num_display} Error processing acts for PRS: {e}", exc_info=True)
                
                activations["skip_batch"] = True
                test_acc = evaluate(model, test_loader, config.get("device", "cpu"))
                activations["skip_batch"] = False

                logger.info(f"Warm-up E{epoch_num_display}/{warmup_ep} CELoss:{avg_epoch_ce_loss:.4f} TrAcc:{train_acc:.2f}% TeAcc:{test_acc:.2f}% PRS:{current_epoch_prs:.4f}")
                scheduler.step()

                if epoch_num_display == warmup_ep and warmup_ep > 0 :
                    logger.info(f"Warm-up E{epoch_num_display} End. Computing initial major regions...")
                    if activations_from_last_full_epoch_for_mrv and labels_from_last_full_epoch_for_mrv:
                        acts_np_warmup_final = torch.cat(activations_from_last_full_epoch_for_mrv).cpu().numpy()
                        lbls_np_warmup_final = np.concatenate(labels_from_last_full_epoch_for_mrv)
                        
                        if len(acts_np_warmup_final) == len(lbls_np_warmup_final) and len(acts_np_warmup_final) > 0:
                            major_regions, unique_patterns_warmup = compute_major_regions(
                                acts_np_warmup_final, lbls_np_warmup_final, num_classes=num_classes, logger=logger
                            )
                            logger.info(f"Warm-up End. Computed MRs. Num classes in MR: {len(major_regions) if major_regions else 'None'}")
                            if epoch_num_display in save_ep:
                                save_model_checkpoint(
                                    model=model, optimizer=optimizer, scheduler=scheduler,
                                    modelname=modelname, dataset_name=dataset_name, batch_size=batch_size,
                                    metrics={"epoch": [epoch_num_display], "test_accuracy": [test_acc], "prs_ratios": [current_epoch_prs]},
                                    logger=logger, config=config, epoch=epoch_num_display, prs_enabled=False,
                                    major_regions=major_regions, unique_patterns=unique_patterns_warmup
                                )
                        else:
                            logger.error(f"Warm-up End. Mismatch/empty for MRV comp. Acts:{len(acts_np_warmup_final)}, Lbls:{len(lbls_np_warmup_final)}.")
                    else:
                        logger.error("Warm-up End. Cannot compute MRs: no acts or labels from last warm-up epoch.")
            
            if not resumed and major_regions is None: # Should have been set if warmup_ep > 0
                 logger.error("[Post Warm-up] 'major_regions' is still None and not resumed. Check warm-up MR computation.")
        
        if major_regions:
            logger.info(f"[Pre-PRS Stage] Using MRs. Num classes with MRVs: {len(major_regions)}.")
        elif config.get("lambda_mrv",0) > 0 or config.get("lambda_hamming",0) > 0: # Only warn if PRS losses are active
            logger.warning("[Pre-PRS Stage] 'major_regions' is None. PRS regularization will be zero/ineffective.")

        start_prs_epoch = warmup_ep
        if warmup_ep > 0 or (resumed and major_regions is not None): 
            logger.info(f"[PRS Setup] Freezing final layer for '{modelname}'...")
            freeze_final_layer(model, modelname, logger)
            
            logger.debug("[DEBUG] Final layer parameters freeze status:")
            for name, param in model.named_parameters():
                if 'classifier.6' in name:
                    logger.debug(f"  Param: {name}, requires_grad: {param.requires_grad}")

            
            unfrozen_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(unfrozen_params if unfrozen_params else model.parameters(), # Fallback if all frozen
                                   lr=config.get("learning_rate_prs", config.get("learning_rate", 0.001)))
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("lr_step_size_prs", scheduler_step_size),
                gamma=config.get("lr_gamma_prs", scheduler_gamma)
            )
            logger.info(f"[PRS Setup] Optimizer/Scheduler re-init for PRS. Unfrozen params: {len(unfrozen_params)}. LR: {optimizer.param_groups[0]['lr']:.2e}")
        else: 
            logger.info("[PRS Setup] No warm-up & not resumed / no MRs. Training all layers from scratch with PRS if lambdas > 0.")
            # Optimizer and scheduler from before are used.

        metrics_prs = {
            "epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": [],
            "mrv_loss": [], "hamming_loss": [], "ce_loss_prs": []
        }
        
        logger.info(f"===== PRS-REGULARISED TRAINING STAGE (Epochs {start_prs_epoch+1} to {total_ep}) =====")
        for epoch in range(start_prs_epoch, total_ep):
            epoch_num_display = epoch + 1
            logger.info(f"--- PRS Epoch {epoch_num_display}/{total_ep} (Overall {epoch_num_display}) ---")
            if config.get("recompute_mr", False):
                if activations_from_last_full_epoch_for_mrv and labels_from_last_full_epoch_for_mrv:
                    logger.debug(f"PRS E{epoch_num_display} MRV Recomp: Using {len(activations_from_last_full_epoch_for_mrv)} act batches & {len(labels_from_last_full_epoch_for_mrv)} label batches from prev epoch.")
                    try:
                        if not activations_from_last_full_epoch_for_mrv or not all(isinstance(t, torch.Tensor) for t in activations_from_last_full_epoch_for_mrv):
                            raise ValueError("Prev epoch activations list empty or contains non-tensors.")
                        
                        acts_np_prev_epoch = torch.cat(activations_from_last_full_epoch_for_mrv).cpu().numpy()
                        lbls_np_prev_epoch = np.concatenate(labels_from_last_full_epoch_for_mrv)
                        
                        logger.debug(f"[PRS Epoch {epoch_num_display}] Prev acts shape: {acts_np_prev_epoch.shape}")
                        logger.debug(f"[PRS Epoch {epoch_num_display}] Label set: {np.unique(lbls_np_prev_epoch)}")
                        
                        logger.info(f"PRS E{epoch_num_display} MRV Recomp: Acts samples: {len(acts_np_prev_epoch)}, Labels samples: {len(lbls_np_prev_epoch)}")

                        if len(acts_np_prev_epoch) == len(lbls_np_prev_epoch) and len(acts_np_prev_epoch) > 0:
                            major_regions, _ = compute_major_regions(
                                acts_np_prev_epoch, lbls_np_prev_epoch, num_classes=num_classes, logger=logger
                            )
                            logger.info(f"PRS E{epoch_num_display} Recomputed MRs. Num classes in MR: {len(major_regions) if major_regions else 'None'}")
                            if major_regions:
                                for c, info in major_regions.items():
                                    mrv_vec = np.array(info["mrv"])
                                    # take the first 10 non-zero elements (or fewer if vector is sparse)
                                    nz_vals   = mrv_vec[mrv_vec != 0]
                                    preview   = nz_vals[:10] if nz_vals.size else []
                                    logger.info(
                                        f"[MRV-Preview][cls={c:2}] len={len(mrv_vec):4d} "
                                        f"non-zeros={nz_vals.size:4d}  first_nz={preview}"
                                    )
                        else:
                            logger.warning(f"PRS E{epoch_num_display} Mismatch/empty for MR recomp. Acts:{len(acts_np_prev_epoch)}, Lbls:{len(lbls_np_prev_epoch)}. Using previous MRs.")
                    except Exception as e:
                        logger.error(f"PRS E{epoch_num_display} Error recomputing MRs: {e}. Using previous MRs.", exc_info=True)
                elif major_regions is None and not resumed : 
                    logger.warning(f"PRS E{epoch_num_display} No acts/labels from prev epoch AND no initial MRs. MRV/HAM loss will be 0.")
                elif epoch > start_prs_epoch : # Not first PRS epoch, but still missing data
                    logger.warning(f"PRS E{epoch_num_display} No acts or labels from previous PRS epoch to recompute MRs. Using existing MRs.")

            model.train()
            current_epoch_batch_activations = [] # Reset for the current PRS epoch
            current_epoch_batch_labels = []      # Reset for the current PRS epoch
            activations["skip_batch"] = False

            current_epoch_total_ce_loss = 0.0; current_epoch_total_mrv_loss = 0.0; current_epoch_total_ham_loss = 0.0; current_epoch_total_combined_loss = 0.0
            correct_preds_epoch = 0; total_samples_epoch = 0

            for batch_idx, (inp, lbl) in enumerate(train_loader):
                activations["current"] = None
                inp, lbl = inp.to(config.get("device", "cpu")), lbl.to(config.get("device", "cpu"))
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=config.get("device", "cpu").split(':')[0], enabled=use_amp):
                    out = model(inp)
                    logits = out.detach()
                    if not torch.isfinite(logits).all():
                        logger.error("[DEBUG] Non-finite logits detected before CE loss!")
                    logger.debug(f"[DEBUG] Logits stats - min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")

                    ce_loss_batch = criterion(out, lbl)

                if not torch.isfinite(ce_loss_batch):
                    logger.error(f"[PRS E{epoch_num_display} B{batch_idx+1}] CE loss NaN/Inf. Skipping batch.")
                    continue

                # Early activation check
                if activations["current"] is None:
                    logger.warning(f"[PRS E{epoch_num_display} B{batch_idx+1}] Activation is None. Skipping batch.")
                    continue
                else:
                    acts = activations["current"].detach()
                    logger.debug(f"[DEBUG] Activations stats - min: {acts.min().item():.4f}, max: {acts.max().item():.4f}, mean: {acts.mean().item():.4f}")
                    if not torch.isfinite(acts).all():
                        logger.error("[DEBUG] Non-finite activations detected!")

                combined_loss_batch = config.get('lambda_ce', 1.0) * ce_loss_batch
                mrv_loss_batch = torch.tensor(0.0, device=combined_loss_batch.device)
                ham_loss_batch = torch.tensor(0.0, device=combined_loss_batch.device)

                if major_regions:
                    mrv_loss_batch = compute_mrv_loss(activations["current"], lbl, major_regions, logger)
                    ham_loss_batch = compute_hamming_loss(activations["current"], lbl, major_regions, logger)

                    logger.debug(f"[Losses B{batch_idx+1}] CE={ce_loss_batch.item():.4f} MRV={mrv_loss_batch.item():.4f} HAM={ham_loss_batch.item():.4f}")

                    if not torch.isfinite(mrv_loss_batch): mrv_loss_batch = torch.tensor(0.0, device=mrv_loss_batch.device)
                    if not torch.isfinite(ham_loss_batch): ham_loss_batch = torch.tensor(0.0, device=ham_loss_batch.device)

                    combined_loss_batch += config.get("lambda_mrv", 0.0) * mrv_loss_batch + config.get("lambda_hamming", 0.0) * ham_loss_batch

                if not torch.isfinite(combined_loss_batch):
                    logger.error(f"[Loss B{batch_idx+1}] Combined loss NaN/Inf. Skipping batch.")
                    continue

                scaler.scale(combined_loss_batch).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if not np.isfinite(grad_norm):
                            logger.error(f"[Gradient NaN] {name} has invalid grad norm!")
                        elif grad_norm > 100:
                            logger.warning(f"[High Gradient] {name} grad norm={grad_norm:.2f}")



                scaler.step(optimizer)
                scaler.update()

                # Update metrics
                current_epoch_total_ce_loss += ce_loss_batch.item() * inp.size(0)
                current_epoch_total_mrv_loss += mrv_loss_batch.item() * inp.size(0)
                current_epoch_total_ham_loss += ham_loss_batch.item() * inp.size(0)
                current_epoch_total_combined_loss += combined_loss_batch.item() * inp.size(0)
                correct_preds_epoch += (out.argmax(1) == lbl).sum().item()
                total_samples_epoch += lbl.size(0)

                # Save activations for MRV recomputation
                current_epoch_batch_labels.append(lbl.cpu().numpy())
                current_epoch_batch_activations.append(activations["current"].detach().cpu())

            # --- END OF PRS EPOCH BATCH LOOP ---

            labels_from_last_full_epoch_for_mrv = current_epoch_batch_labels
            activations_from_last_full_epoch_for_mrv = current_epoch_batch_activations
            
            # ... (Calculate average losses, PRS, test_acc as before) ...
            avg_ce_loss_prs_epoch = current_epoch_total_ce_loss / total_samples_epoch if total_samples_epoch > 0 else 0
            avg_mrv_loss_epoch = current_epoch_total_mrv_loss / total_samples_epoch if total_samples_epoch > 0 else 0
            avg_ham_loss_epoch = current_epoch_total_ham_loss / total_samples_epoch if total_samples_epoch > 0 else 0
            avg_combined_loss_epoch = current_epoch_total_combined_loss / total_samples_epoch if total_samples_epoch > 0 else 0
            train_acc_epoch = 100 * correct_preds_epoch / total_samples_epoch if total_samples_epoch > 0 else 0

            current_epoch_prs_value = 0.0
            if activations_from_last_full_epoch_for_mrv: # Use the newly collected activations
                try:
                    if not activations_from_last_full_epoch_for_mrv or not all(isinstance(t, torch.Tensor) for t in activations_from_last_full_epoch_for_mrv):
                        logger.error(f"PRS E{epoch_num_display} Invalid acts_from_last_epoch for PRS calc.")
                    else:
                        acts_np_prs_epoch = torch.cat(activations_from_last_full_epoch_for_mrv).cpu().numpy()
                        if len(acts_np_prs_epoch) > 0:
                            unique_acts_count_prs = compute_unique_activations(acts_np_prs_epoch, logger)
                            current_epoch_prs_value = unique_acts_count_prs / len(train_ds) if len(train_ds) > 0 else 0.0
                            logger.debug(f"[PRS Metric] Unique activations: {unique_acts_count_prs}, Dataset size: {len(train_ds)}, PRS ratio: {current_epoch_prs_value:.4f}")
                except Exception as e: logger.error(f"PRS E{epoch_num_display} Error processing acts for PRS: {e}", exc_info=True)
            
            activations["skip_batch"] = True
            test_acc_epoch = evaluate(model, test_loader, config.get("device", "cpu"))
            activations["skip_batch"] = False

            logger.info(f"PRS E{epoch_num_display}/{total_ep} CombLoss:{avg_combined_loss_epoch:.4f} (CE:{avg_ce_loss_prs_epoch:.4f} MRV:{avg_mrv_loss_epoch:.4f} HAM:{avg_ham_loss_epoch:.4f}) TrAcc:{train_acc_epoch:.2f}% TeAcc:{test_acc_epoch:.2f}% PRS:{current_epoch_prs_value:.4f}")
            scheduler.step()

            # ... (Record metrics, save checkpoint as before) ...
            metrics_prs["epoch"].append(epoch_num_display); metrics_prs["train_accuracy"].append(train_acc_epoch)
            metrics_prs["test_accuracy"].append(test_acc_epoch); metrics_prs["prs_ratios"].append(current_epoch_prs_value)
            metrics_prs["ce_loss_prs"].append(avg_ce_loss_prs_epoch); metrics_prs["mrv_loss"].append(avg_mrv_loss_epoch)
            metrics_prs["hamming_loss"].append(avg_ham_loss_epoch)

            if epoch_num_display in save_ep:
                logger.info(f"PRS E{epoch_num_display} Designated save epoch. Saving checkpoint...")
                unique_patterns_to_save = None
                if activations_from_last_full_epoch_for_mrv and labels_from_last_full_epoch_for_mrv:
                    try:
                        acts_for_save = torch.cat(activations_from_last_full_epoch_for_mrv).cpu().numpy()
                        lbls_for_save = np.concatenate(labels_from_last_full_epoch_for_mrv)
                        if len(acts_for_save) == len(lbls_for_save) and len(acts_for_save) > 0:
                            _, unique_patterns_to_save = compute_major_regions(acts_for_save, lbls_for_save, num_classes=num_classes, logger=logger)
                    except Exception as e: logger.error(f"PRS E{epoch_num_display} Save. Error computing unique_patterns: {e}", exc_info=True)
                
                save_model_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler, modelname=modelname, dataset_name=dataset_name,
                    batch_size=batch_size, metrics=metrics_prs, logger=logger, config=config, epoch=epoch_num_display,
                    prs_enabled=True, major_regions=major_regions, unique_patterns=unique_patterns_to_save, extra_tag="recomputation"
                )
        
        if hook_handle: hook_handle.remove()
        results[f"{modelname}_{dataset_name}_batch_{batch_size}"] = metrics_prs
        logger.info(f"===== Finished Training Run: Model={modelname}, Dataset={dataset_name}, BatchSize={batch_size} =====")

    logger.info("========== ALL TRAINING RUNS COMPLETE ==========")

if __name__ == "__main__":
    train()