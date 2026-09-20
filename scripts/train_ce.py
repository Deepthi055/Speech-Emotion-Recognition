import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.embedding_dataset import EmbeddingDataset
from src.models.baseline import WavLMClassifier
from src.utils.logging import setup_logger
from src.utils.seed import set_seed
from src.datasets.metadata import EMOTION_LABELS

logger = setup_logger("SER_Baseline")

def train_one_epoch(model, dataloader, optimizer, criterion, device, dry_run=False):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        if dry_run and step >= 2:
            break
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
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

        logits = model(embeddings)
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

def main():
    parser = argparse.ArgumentParser(description="Train CPU-only WavLM + Cross-Entropy Baseline Classifier")
    parser.add_argument("--train_npz", type=str, default="data/embeddings/wavlm_mean_train.npz", help="Path to train cached embeddings .npz")
    parser.add_argument("--val_npz", type=str, default="data/embeddings/wavlm_mean_val.npz", help="Path to val cached embeddings .npz")
    parser.add_argument("--test_npz", type=str, default="data/embeddings/wavlm_mean_test.npz", help="Path to test cached embeddings .npz")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--output_dir", type=str, default="runs/wavlm_ce_baseline", help="Directory to save checkpoints and metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Perform 2-step dry run for testing")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")
    logger.info(f"Running baseline training on device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir))

    max_samples = 32 if args.dry_run else None

    logger.info(f"Loading cached train embeddings from {args.train_npz}...")
    train_ds = EmbeddingDataset(args.train_npz, max_samples=max_samples)
    val_ds = EmbeddingDataset(args.val_npz, max_samples=max_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = WavLMClassifier(in_dim=768, num_classes=len(EMOTION_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epochs = 2 if args.dry_run else args.epochs
    patience_counter = 0
    best_uar = 0.0

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, dry_run=args.dry_run)
        val_metrics = evaluate_model(model, val_loader, device, dry_run=args.dry_run)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("UAR/val", val_metrics["uar"], epoch)
        writer.add_scalar("F1_Macro/val", val_metrics["f1_macro"], epoch)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val UAR: {val_metrics['uar']:.4f} | "
            f"Val F1 (Macro): {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["uar"] > best_uar:
            best_uar = val_metrics["uar"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"Saved new best model checkpoint (Val UAR: {best_uar:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience and not args.dry_run:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    # Final Evaluation on Test Set if test_npz exists
    test_npz_path = Path(args.test_npz)
    if test_npz_path.exists():
        logger.info(f"Loading best model for test evaluation from {args.test_npz}...")
        best_model_path = output_dir / "best_model.pt"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_ds = EmbeddingDataset(args.test_npz, max_samples=max_samples)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        test_metrics = evaluate_model(model, test_loader, device, dry_run=args.dry_run)

        logger.info("\n=== TEST SET CLASSIFICATION REPORT ===")
        logger.info("\n" + test_metrics["classification_report_text"])

        results = {
            "best_val_uar": float(best_uar),
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

        logger.info(f"Final Test Metrics -> Acc: {test_metrics['accuracy']:.4f} | UAR: {test_metrics['uar']:.4f} | F1 (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"Results saved cleanly to {output_dir / 'metrics.json'}")

    writer.close()
    logger.info("Training process completed.")

if __name__ == "__main__":
    main()
