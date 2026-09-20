import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.embedding_dataset import EmbeddingDataset
from src.datasets.metadata import EMOTION_LABELS
from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel
from src.losses.supervised_contrastive import SupConLoss
from src.losses.contrastive_weights import (
    compute_standard_weights,
    compute_speaker_aware_weights,
    compute_corpus_aware_weights,
    compute_speaker_corpus_aware_weights,
)
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("SER_Trainer")


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_contrastive_weights(variant: str, labels: torch.Tensor, speaker_ids: torch.Tensor, corpus_ids: torch.Tensor):
    if variant == "standard_supcon":
        return compute_standard_weights(labels)
    elif variant == "speaker_supcon":
        return compute_speaker_aware_weights(labels, speaker_ids)
    elif variant == "corpus_supcon":
        return compute_corpus_aware_weights(labels, corpus_ids)
    elif variant == "proposed_supcon":
        return compute_speaker_corpus_aware_weights(labels, speaker_ids, corpus_ids)
    else:
        raise ValueError(f"Unknown supcon variant: {variant}")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    ce_criterion,
    supcon_criterion,
    variant,
    supcon_weight,
    ce_weight,
    device,
    dry_run=False,
):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        if dry_run and step >= 2:
            break

        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        speaker_ids = batch["speaker_id"].to(device)
        corpus_ids = batch["corpus_id"].to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        logits = outputs["logits"]
        projections = outputs["projections"]

        if variant == "ce":
            loss = ce_criterion(logits, labels)
        else:
            weights_mask = get_contrastive_weights(variant, labels, speaker_ids, corpus_ids)
            sup_loss = supcon_criterion(projections, weights_mask=weights_mask)
            ce_loss = ce_criterion(logits, labels)
            loss = supcon_weight * sup_loss + ce_weight * ce_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    num_steps = min(len(dataloader), 2) if dry_run else len(dataloader)
    return total_loss / max(num_steps, 1)


@torch.no_grad()
def evaluate_model(model, dataloader, device, dry_run=False):
    model.eval()
    all_preds = []
    all_labels = []

    for step, batch in enumerate(dataloader):
        if dry_run and step >= 2:
            break

        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)

        outputs = model(embeddings)
        logits = outputs["logits"]
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(EMOTION_LABELS))))

    report = classification_report(
        all_labels,
        all_preds,
        target_names=EMOTION_LABELS,
        labels=list(range(len(EMOTION_LABELS))),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=EMOTION_LABELS,
        labels=list(range(len(EMOTION_LABELS))),
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "uar": float(uar),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report_dict": report,
        "classification_report_text": report_text,
    }


def run_experiment(config_path: str | Path, dry_run: bool = False, seed: int = 42):
    set_seed(seed)
    config = load_config(config_path)

    exp_name = config["experiment"]["name"]
    variant = config["experiment"]["variant"]
    device = torch.device("cpu")

    logger.info(f"\n==================================================")
    logger.info(f"Running Experiment: {exp_name} (Variant: {variant})")
    logger.info(f"==================================================")

    train_cfg = config["training"]
    data_cfg = config["data"]
    loss_cfg = config.get("loss", {})

    output_dir = Path(train_cfg.get("output_dir", f"runs/{exp_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir))

    max_samples = 32 if dry_run else None
    train_ds = EmbeddingDataset(data_cfg["train_npz"], max_samples=max_samples)
    val_ds = EmbeddingDataset(data_cfg["val_npz"], max_samples=max_samples)
    test_ds = EmbeddingDataset(data_cfg["test_npz"], max_samples=max_samples)

    batch_size = train_cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = WavLMEmbeddingSupConModel(
        in_dim=config["model"].get("in_dim", 768),
        proj_dim=config["model"].get("proj_dim", 128),
        num_classes=config["model"].get("num_classes", 6),
        use_projection=(variant != "ce"),
        use_classifier=True,
    ).to(device)

    ce_criterion = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=loss_cfg.get("temperature", 0.07))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = 2 if dry_run else train_cfg.get("epochs", 30)
    patience = train_cfg.get("patience", 7)
    patience_counter = 0
    best_val_uar = 0.0

    supcon_weight = float(train_cfg.get("supcon_weight", 1.0))
    ce_weight = float(train_cfg.get("ce_weight", 1.0))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ce_criterion,
            supcon_criterion,
            variant,
            supcon_weight,
            ce_weight,
            device,
            dry_run=dry_run,
        )
        val_metrics = evaluate_model(model, val_loader, device, dry_run=dry_run)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("UAR/val", val_metrics["uar"], epoch)
        writer.add_scalar("F1_Macro/val", val_metrics["f1_macro"], epoch)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val UAR: {val_metrics['uar']:.4f} | "
            f"Val F1 (Macro): {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["uar"] > best_val_uar:
            best_val_uar = val_metrics["uar"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f" -> Saved new best model checkpoint (Val UAR: {best_val_uar:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and not dry_run:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    # Evaluate Best Model on Test Set
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate_model(model, test_loader, device, dry_run=dry_run)

    results = {
        "experiment": exp_name,
        "variant": variant,
        "best_val_uar": float(best_val_uar),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_uar": float(test_metrics["uar"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
        "test_f1_weighted": float(test_metrics["f1_weighted"]),
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "classification_report": test_metrics["classification_report_dict"],
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save(output_dir / "confusion_matrix.npy", test_metrics["confusion_matrix"])

    logger.info(f"\n--- TEST METRICS ({exp_name}) ---")
    logger.info(
        f"Test Accuracy (WAR): {results['test_accuracy']:.4f} | "
        f"Test UAR: {results['test_uar']:.4f} | "
        f"Macro-F1: {results['test_f1_macro']:.4f}"
    )
    logger.info(f"Saved checkpoint and metrics to {output_dir}\n")

    writer.close()
    return results


def summarize_all_results():
    exp_dirs = [
        ("WavLM + CE (Baseline)", "runs/wavlm_ce_baseline"),
        ("Standard SupCon", "runs/wavlm_supcon"),
        ("Speaker-Aware SupCon", "runs/wavlm_speaker_supcon"),
        ("Corpus-Aware SupCon (Step 9)", "runs/wavlm_corpus_supcon"),
        ("Speaker + Corpus-Aware SupCon (Step 10 Proposed)", "runs/wavlm_proposed_supcon"),
    ]

    summary = []
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY COMPARISON TABLE ACROSS ALL EXPERIMENTS")
    logger.info("=" * 80)
    header = f"{'Experiment':<45} | {'Acc (WAR)':<10} | {'UAR':<10} | {'Macro-F1':<10}"
    logger.info(header)
    logger.info("-" * 80)

    for label, rdir in exp_dirs:
        mfile = Path(rdir) / "metrics.json"
        if mfile.exists():
            with open(mfile, "r") as f:
                res = json.load(f)
            acc = res.get("test_accuracy", 0.0)
            uar = res.get("test_uar", 0.0)
            f1 = res.get("test_f1_macro", 0.0)
            summary.append({
                "method": label,
                "dir": rdir,
                "test_accuracy": acc,
                "test_uar": uar,
                "test_f1_macro": f1,
            })
            logger.info(f"{label:<45} | {acc:<10.4f} | {uar:<10.4f} | {f1:<10.4f}")
        else:
            logger.info(f"{label:<45} | N/A        | N/A        | N/A")

    logger.info("=" * 80 + "\n")

    Path("runs").mkdir(exist_ok=True)
    with open("runs/comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Unified CPU Trainer for Speech Emotion Recognition Experiments")
    parser.add_argument("--config", type=str, help="Path to specific YAML config file")
    parser.add_argument("--all", action="store_true", help="Run all 5 experiment configurations sequentially")
    parser.add_argument("--dry-run", action="store_true", help="Perform 2-step dry run across experiments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    configs = [
        "configs/wavlm_ce_baseline.yaml",
        "configs/wavlm_supcon.yaml",
        "configs/wavlm_speaker_supcon.yaml",
        "configs/wavlm_corpus_supcon.yaml",
        "configs/wavlm_proposed_supcon.yaml",
    ]

    if args.all:
        for cfg in configs:
            run_experiment(cfg, dry_run=args.dry_run, seed=args.seed)
        summarize_all_results()
    elif args.config:
        run_experiment(args.config, dry_run=args.dry_run, seed=args.seed)
    else:
        logger.error("Please specify either --config <path> or --all to run training.")


if __name__ == "__main__":
    main()
