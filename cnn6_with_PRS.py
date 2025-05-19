import os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
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
    ckpt_dir = os.path.join("models", "saved",
                            f"{modelname}_{dataset_name}_batch_{batch_size}")
    warm_ep  = config["warmup_epochs"]
    ep_dir   = os.path.join(ckpt_dir, f"epoch_{warm_ep}")

    if not os.path.exists(ep_dir):
        logger.warning("Warm‑up checkpoint not found – starting from scratch.")
        return False, model, optimizer, None, None, None

    logger.info(f"Loading warm‑up checkpoint: {ep_dir}")
    try:
        model.load_state_dict(torch.load(os.path.join(ep_dir, "model.pth"),
                                         map_location=config["device"]))
        optimizer.load_state_dict(torch.load(os.path.join(ep_dir, "optimizer.pth"),
                                             map_location=config["device"]))
        sched_state = (torch.load(os.path.join(ep_dir, "scheduler.pth"),
                                  map_location=config["device"])
                       if os.path.exists(os.path.join(ep_dir, "scheduler.pth"))
                       else None)
        with open(os.path.join(ep_dir, "major_regions.json")) as f:
            major_regions = json.load(f)
        with open(os.path.join(ep_dir, "unique_patterns.json")) as f:
            unique_patterns = json.load(f)

        model.to(config["device"]).train()
        return True, model, optimizer, sched_state, major_regions, unique_patterns
    except Exception as e:
        logger.error(f"Checkpoint load failed: {e}")
        return False, model, optimizer, None, None, None

# ──────────────────────────────────────────────────────────────────────────
#  main routine
# ──────────────────────────────────────────────────────────────────────────
def train():
    set_seed(config["seed"])
    results = {}

    for modelname in tqdm(config["models"],  desc="Model"):
      for dataset_name in tqdm(config["datasets"], desc="Dataset"):
       for batch_size   in tqdm(config["batch_sizes"], desc="Batch size"):

        # ───────── logger
        logger = setup_logger(modelname, dataset_name, batch_size)
        logger.info(f"[CONFIG] λ_mrv={config['lambda_mrv']} "
                    f"λ_hamming={config['lambda_hamming']}")

        # ───────── data
        train_ds, test_ds, in_ch = get_datasets(dataset_name)
        train_loader = DataLoader(
            train_ds, batch_size, shuffle=True,
            worker_init_fn=lambda _: np.random.seed(config["seed"]),
            generator=torch.Generator().manual_seed(config["seed"])
        )
        test_loader  = DataLoader(test_ds, config["test_batch_size"], shuffle=False)

        # ───────── model & optimiser
        model = get_model(modelname, in_ch).to(config["device"])
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        scaler    = torch.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

        # ───────── hook (state stays the entire run)
        activations = {"penultimate": [], "skip_batch": False, "current": None}
        hook_handle = register_activation_hook(
            model, activations, modelname, dataset_name, batch_size, logger
        )

        # ───────── resume warm‑up?
        warmup_ep = config["warmup_epochs"]
        total_ep   = config["epochs"]
        save_ep   = {*range(warmup_ep, total_ep + 1, 50), total_ep}

        resumed, model, optimizer, sched_state, major_regions, _ = load_checkpoint_if_exists(
            model, optimizer, modelname, dataset_name, batch_size, logger
        )
        if sched_state:
            scheduler.load_state_dict(sched_state)

        if not resumed:
            logger.info("=== WARM‑UP ===")
            for epoch in range(warmup_ep):
                model.train()
                activations["penultimate"].clear()
                activations["skip_batch"] = False
                epoch_loss = correct = total = 0
                batch_labels = []

                for inp, lbl in train_loader:
                    inp, lbl = inp.to(config["device"]), lbl.to(config["device"])
                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(device_type="cuda"
                                            if torch.cuda.is_available() else "cpu"):
                        out  = model(inp)
                        loss = criterion(out, lbl)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    correct    += (out.argmax(1) == lbl).sum().item()
                    total      += lbl.size(0)
                    batch_labels.append(lbl.cpu().numpy())

                if activations["penultimate"]:
                    acts_np = torch.cat(activations["penultimate"]).numpy()
                    prs     = compute_unique_activations(acts_np, logger) / len(train_ds)
                else:
                    prs = 0.0
                    logger.warning("No activations captured this epoch.")

                activations["skip_batch"] = True
                test_acc = evaluate(model, test_loader, config["device"])
                activations["skip_batch"] = False

                logger.info(f"[Warm‑up {epoch+1}/{warmup_ep}] "
                            f"loss={epoch_loss:.4f} "
                            f"train_acc={100*correct/total:.2f}% "
                            f"test_acc={test_acc:.2f}% "
                            f"PRS={prs:.4f}")
                scheduler.step()

            # build MRV once after warm‑up
            acts_np = torch.cat(activations["penultimate"]).numpy()
            lbls_np = np.concatenate(batch_labels)
            major_regions, _ = compute_major_regions(
                acts_np, lbls_np, num_classes=10, logger=logger
            )

        # ───────── freeze final layer
        start_prs = warmup_ep
        freeze_final_layer(model, modelname, logger)
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=config["learning_rate"])
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_step_size"] if "lr_step_size" in config else 10,
            gamma=config["lr_gamma"]     if "lr_gamma"     in config else 0.1
        )
        
        metrics = {
            "epoch": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "prs_ratios": []
        }

        prev_labels = []
        activations["penultimate"].clear()  

        logger.info("=== PRS‑regularised stage ===")
        for epoch in range(start_prs, total_ep):
            # ─── Recompute MRV & regions from last‐epoch activations ───
            if activations["penultimate"]:
                acts_np  = torch.cat(activations["penultimate"]).cpu().numpy()
                lbls_np  = np.concatenate(prev_labels)
                major_regions, _ = compute_major_regions(
                    acts_np, lbls_np, num_classes=10, logger=logger
                )
            else:
                logger.warning(f"[Epoch {epoch+1}] No penultimate activations to recompute regions.")

            model.train()
            activations["penultimate"].clear()
            activations["skip_batch"] = False

            ep_loss = ep_mrv = ep_ham = correct = total = 0
            curr_labels = []

            for inp, lbl in train_loader:
                activations["current"] = None
                inp, lbl = inp.to(config["device"]), lbl.to(config["device"])
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda"
                                        if torch.cuda.is_available() else "cpu"):
                    out  = model(inp)
                    ce   = criterion(out, lbl)

                loss  = ce
                preds = out.argmax(1)

                if activations["current"] is not None:
                    mrv = compute_mrv_loss(
                        activations["current"], lbl, preds, major_regions, logger)
                    ham = compute_hamming_loss(
                        activations["current"], lbl, preds, major_regions)

                    ep_mrv += mrv.item()
                    ep_ham += ham.item()
                    loss   += (config["lambda_mrv"] * mrv +
                               config["lambda_hamming"] * ham)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                ep_loss += loss.item()
                correct += (preds == lbl).sum().item()
                total   += lbl.size(0)

                if activations["current"] is not None:
                    activations["penultimate"].append(
                        activations["current"].detach().cpu())
                curr_labels.append(lbl.cpu().numpy())

            train_acc = 100 * correct / total

            if activations["penultimate"]:
                acts_np = torch.cat(activations["penultimate"]).numpy()
                prs     = compute_unique_activations(acts_np, logger) / len(train_ds)
            else:
                prs = 0.0
                logger.warning("No activations captured this epoch.")

            activations["skip_batch"] = True
            test_acc = evaluate(model, test_loader, config["device"])
            activations["skip_batch"] = False

            logger.info(f"[{epoch+1}/{total_ep}] "
                        f"loss={ep_loss:.4f} "
                        f"train_acc={train_acc:.2f}% "
                        f"test_acc={test_acc:.2f}% "
                        f"PRS={prs:.4f} "
                        f"mrv={ep_mrv/len(train_loader):.4f} "
                        f"ham={ep_ham/len(train_loader):.4f}")
            scheduler.step()

            # ✱record metrics✱
            metrics["epoch"].append(epoch + 1)
            metrics["train_accuracy"].append(train_acc)
            metrics["test_accuracy"].append(test_acc)
            metrics["prs_ratios"].append(prs)

            # save checkpoint every 50 epochs & on the last one
            if epoch in save_ep and activations["penultimate"]:
                acts_np = torch.cat(activations["penultimate"]).numpy()
                lbls_np = np.concatenate(curr_labels)
                major_regions, unique_patterns = compute_major_regions(
                    acts_np, lbls_np, 10, logger=logger)

                save_model_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    modelname=modelname,
                    dataset_name=dataset_name,
                    batch_size=batch_size,
                    metrics=metrics,            # ← real metrics dict
                    logger=logger,
                    config=config,              # ← the imported config dict
                    epoch=epoch,
                    prs_enabled=True,
                    major_regions=major_regions,
                    unique_patterns=unique_patterns
                )
            prev_labels = curr_labels

        hook_handle.remove()   # tidy up
        results[f"{dataset_name}_batch_{batch_size}"] = metrics

    logger.info("=== Training Complete ===")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
