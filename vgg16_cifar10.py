import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from utils import (
    get_datasets, evaluate, compute_unique_activations,
    register_activation_hook, compute_major_regions,
    save_model_checkpoint, set_seed
)
from utils.logger import setup_logger
from config import config

# ────────────────────────────────────────────────────────────────────
def train():
    set_seed(config["seed"])
    results, base_lr = {}, 1e-3     # ← LR that works for VGG & CNN

    for modelname in tqdm(config["models"], desc="Model"):
      for dataset in tqdm(config["datasets"], desc="Dataset"):
       for batch in tqdm(config["batch_sizes"], desc="Batch"):

        logger = setup_logger(modelname, dataset, batch)
        logger.info(f"lr={base_lr}, weight_decay=0")

        # data ----------------------------------------------------------------
        train_ds, test_ds, in_ch = get_datasets(dataset)
        train_loader = DataLoader(
            train_ds, batch, shuffle=True,
            worker_init_fn=lambda _: np.random.seed(config["seed"]),
            generator=torch.Generator().manual_seed(config["seed"])
        )
        test_loader  = DataLoader(test_ds, config["test_batch_size"], shuffle=False)

        # model ----------------------------------------------------------------
        model = get_model(modelname, in_ch).to(config["device"])
        optimiser = optim.Adam(model.parameters(), lr=base_lr)
        loss_fn   = nn.CrossEntropyLoss()

        # hook state -----------------------------------------------------------
        activations = {"penultimate": [], "skip_batch": False}
        hook = register_activation_hook(model, activations,
                                        modelname, dataset, batch, logger)

        # bookkeeping ----------------------------------------------------------
        metrics = {"epoch": [], "train_accuracy": [], "test_accuracy": [], "prs_ratios": []}
        total_ep  = config["epochs"]
        save_ep   = {*range(50, total_ep+1, 50), total_ep}

        # training loop --------------------------------------------------------
        for epoch in range(1, total_ep+1):
            model.train()
            activations["penultimate"].clear()     # ← keep same list object
            activations["skip_batch"] = False

            ep_loss = correct = total = 0
            batch_lbls = []

            for x, y in train_loader:
                x, y = x.to(config["device"]), y.to(config["device"])

                out  = model(x)
                loss = loss_fn(out, y)

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

                ep_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total   += y.size(0)
                batch_lbls.append(y.cpu().numpy())

            # --- PRS ratio
            if activations["penultimate"]:
                acts = torch.cat(activations["penultimate"]).cpu().numpy()
                lbls = np.concatenate(batch_lbls)
                prs  = compute_unique_activations(acts, logger) / len(train_ds)
            else:
                logger.warning("No activations collected — PRS skipped.")
                prs = 0.0

            # --- validation (keep hook, just skip)
            activations["skip_batch"] = True
            test_acc = evaluate(model, test_loader, config["device"])
            activations["skip_batch"] = False

            train_acc = 100 * correct / total
            logger.info(f"Ep {epoch}/{total_ep} "
                        f"loss={ep_loss:.4f} "
                        f"train={train_acc:.2f}% "
                        f"test={test_acc:.2f}% "
                        f"PRS={prs:.4f}")

            metrics["epoch"].append(epoch)
            metrics["train_accuracy"].append(train_acc)
            metrics["test_accuracy"].append(test_acc)
            metrics["prs_ratios"].append(prs)

            # --- checkpoint
            if epoch in save_ep and activations["penultimate"]:
                major_regions, unique_patterns = compute_major_regions(
                    acts, lbls, num_classes=10, logger=logger)
                save_model_checkpoint(
                    model=model,
                    optimizer=optimiser,
                    modelname=modelname,
                    dataset_name=dataset,
                    batch_size=batch,
                    metrics=metrics,
                    logger=logger,
                    config=config,
                    epoch=epoch,
                    major_regions=major_regions,
                    unique_patterns=unique_patterns,
                    extra_tag="",
                )

        hook.remove()
        results[f"{modelname}_{dataset}_batch_{batch}"] = metrics

    logger.info("Training Complete")
    return results

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()