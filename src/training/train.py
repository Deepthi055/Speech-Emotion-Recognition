import argparse
import json
import os
import numpy as np
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.emobox_dataset import SERDataset, collate_fn
from src.models.ser_model import WavLMSupConModel
from src.losses.supervised_contrastive import SupConLoss
from src.samplers.standard import get_standard_sampler
from src.training.evaluate import evaluate
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

logger = setup_logger()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataloader(dataset, config_sampler, is_train: bool = True):
    sampler_type = config_sampler.get("type", "standard") if is_train else "standard"
    batch_size = config_sampler.get("batch_size", 16)

    if sampler_type == "speaker_corpus_aware" and is_train:
        batch_sampler = SpeakerCorpusAwareBatchSampler(dataset, batch_size=batch_size)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    else:
        batch_sampler = get_standard_sampler(dataset, batch_size=batch_size, shuffle=is_train)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)


def train_one_epoch(model, dataloader, optimizer, ce_criterion, supcon_criterion, variant, weights_config, device, dry_run: bool = False):
    model.train()
    total_loss = 0.0

    supcon_weight = weights_config.get("supcon_weight", 1.0)
    ce_weight = weights_config.get("ce_weight", 1.0)

    for step, batch in enumerate(dataloader):
        if dry_run and step >= 1:  # Only 1 batch in dry-run
            break
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_values, attention_mask=attention_mask)

        loss = torch.tensor(0.0, device=device)

        if variant == "ce":
            logits = outputs["logits"]
            loss = ce_criterion(logits, labels)
        elif variant in ["standard_supcon", "proposed_supcon"]:
            projections = outputs["projections"]
            if variant == "standard_supcon":
                sup_loss = supcon_criterion(projections, labels=labels)
            else:
                speaker_ids = batch["speaker_ids"].to(device)
                corpus_ids = batch["corpus_ids"].to(device)
                weights_mask = compute_speaker_corpus_weights(labels, speaker_ids, corpus_ids)
                sup_loss = supcon_criterion(projections, weights_mask=weights_mask)

            logits = outputs["logits"]
            ce_loss = ce_criterion(logits, labels) if logits is not None else torch.tensor(0.0, device=device)
            loss = supcon_weight * sup_loss + ce_weight * ce_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader) if not dry_run else 2, 1)


def main():
    parser = argparse.ArgumentParser(description="Unified Speech Emotion Recognition Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Run a fast 1-epoch / 1-batch dry run for testing")
    args = parser.parse_args()

    set_seed(42)
    config = load_config(args.config)
    exp_name = config["experiment"]["name"]
    variant = config["experiment"]["variant"]
    logger.info(f"Starting experiment: {exp_name} (variant: {variant})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    save_dir = Path(f"runs/{exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir))

    data_cfg = config["data"]
    # Dry-run: tiny dataset (4 samples), short audio (2s), tiny batch (2)
    # This is just to verify the full pipeline runs end-to-end without errors.
    if args.dry_run:
        max_samples = 4
        max_seconds = 2.0
        dry_batch_size = 2
    else:
        max_samples = None
        max_seconds = data_cfg.get("max_seconds", 6.0)
        dry_batch_size = None

    train_dataset = SERDataset(
        data_cfg["train_csv"],
        target_sr=data_cfg.get("target_sr", 16000),
        max_seconds=max_seconds,
        max_samples=max_samples,
    )
    val_dataset = SERDataset(
        data_cfg["val_csv"],
        target_sr=data_cfg.get("target_sr", 16000),
        max_seconds=max_seconds,
        max_samples=max_samples,
    )

    # Override batch size for dry-run to keep it tiny
    sampler_cfg = dict(config["sampler"])
    if dry_batch_size is not None:
        sampler_cfg["batch_size"] = dry_batch_size

    train_loader = build_dataloader(train_dataset, sampler_cfg, is_train=True)
    val_loader = build_dataloader(val_dataset, sampler_cfg, is_train=False)

    model_cfg = config["model"]
    model = WavLMSupConModel(
        model_id=model_cfg.get("model_id", "microsoft/wavlm-base"),
        num_classes=model_cfg.get("num_classes", 6),
        proj_dim=model_cfg.get("proj_dim", 128),
        use_projection=model_cfg.get("use_projection", True),
        use_classifier=model_cfg.get("use_classifier", True),
        pooling=model_cfg.get("pooling", "attention"),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)

    ce_criterion = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=config.get("loss", {}).get("temperature", 0.07))

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = 1 if args.dry_run else train_cfg.get("epochs", 10)
    patience = train_cfg.get("patience", 5)
    patience_counter = 0
    best_uar = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ce_criterion,
            supcon_criterion,
            variant,
            train_cfg,
            device,
            dry_run=args.dry_run,
        )
        val_metrics = evaluate(model, val_loader, device, dry_run=args.dry_run)
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1_Macro/val", val_metrics["f1_macro"], epoch)
        writer.add_scalar("UAR/val", val_metrics["uar"], epoch)

        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1 (Macro): {val_metrics['f1_macro']:.4f} | "
            f"Val UAR: {val_metrics['uar']:.4f}"
        )
        
        if val_metrics["uar"] > best_uar:
            best_uar = val_metrics["uar"]
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            logger.info(f"New best model saved with UAR: {best_uar:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    torch.save(model.state_dict(), save_dir / "final_model.pt")
    writer.close()

    # Save a final evaluation on validation set with full metrics
    final_metrics = evaluate(model, val_loader, device)
    results = {
        "experiment": exp_name,
        "variant": variant,
        "best_uar": float(best_uar),
        "final_val_uar": float(final_metrics["uar"]),
        "final_val_war": float(final_metrics["accuracy"]),
        "final_val_f1_macro": float(final_metrics["f1_macro"]),
        "final_val_f1_weighted": float(final_metrics["f1_weighted"]),
        "confusion_matrix": final_metrics["confusion_matrix"].tolist(),
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(save_dir / "confusion_matrix.npy", final_metrics["confusion_matrix"])

    logger.info(
        f"Final Results -> UAR: {results['final_val_uar']:.4f} | "
        f"WAR: {results['final_val_war']:.4f} | F1 (Macro): {results['final_val_f1_macro']:.4f}"
    )
    logger.info(f"Results saved to {save_dir / 'results.json'}")
    logger.info(f"Completed experiment: {exp_name}")


if __name__ == "__main__":
    main()



