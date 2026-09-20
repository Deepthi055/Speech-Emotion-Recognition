import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

logger = setup_logger("CrossCorpus_Trainer")

CROSS_CORPUS_SETUPS = {
    "crema_ravdess_to_iemocap": {
        "title": "CREMA-D + RAVDESS -> IEMOCAP",
        "train_corpora": ["CREMA-D", "RAVDESS"],
        "test_corpus": "IEMOCAP",
    },
    "crema_iemocap_to_ravdess": {
        "title": "CREMA-D + IEMOCAP -> RAVDESS",
        "train_corpora": ["CREMA-D", "IEMOCAP"],
        "test_corpus": "RAVDESS",
    },
    "ravdess_iemocap_to_cremad": {
        "title": "RAVDESS + IEMOCAP -> CREMA-D",
        "train_corpora": ["RAVDESS", "IEMOCAP"],
        "test_corpus": "CREMA-D",
    },
}

VARIANTS = [
    ("ce", "WavLM + CE Baseline"),
    ("standard_supcon", "Standard SupCon"),
    ("speaker_supcon", "Speaker-Aware SupCon"),
    ("corpus_supcon", "Corpus-Aware SupCon"),
    ("proposed_supcon", "Speaker + Corpus-Aware SupCon (Proposed)"),
]


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

    return {
        "accuracy": float(acc),
        "uar": float(uar),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report_dict": report,
    }


def run_single_cross_corpus_experiment(
    setup_key: str,
    variant: str,
    variant_title: str,
    dry_run: bool = False,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 7,
):
    set_seed(seed)
    device = torch.device("cpu")
    setup_info = CROSS_CORPUS_SETUPS[setup_key]

    output_dir = Path("results/cross_corpus") / setup_key / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 32 if dry_run else None

    train_npz = "data/embeddings/wavlm_mean_train.npz"
    val_npz = "data/embeddings/wavlm_mean_val.npz"
    test_npz_list = [
        "data/embeddings/wavlm_mean_train.npz",
        "data/embeddings/wavlm_mean_val.npz",
        "data/embeddings/wavlm_mean_test.npz",
    ]

    train_ds = EmbeddingDataset(train_npz, include_corpora=setup_info["train_corpora"], max_samples=max_samples)
    val_ds = EmbeddingDataset(val_npz, include_corpora=setup_info["train_corpora"], max_samples=max_samples)
    test_ds = EmbeddingDataset(test_npz_list, include_corpora=[setup_info["test_corpus"]], max_samples=max_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = WavLMEmbeddingSupConModel(
        in_dim=768,
        proj_dim=128,
        num_classes=6,
        use_projection=(variant != "ce"),
        use_classifier=True,
    ).to(device)

    ce_criterion = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_epochs = 2 if dry_run else epochs
    patience_counter = 0
    best_val_uar = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ce_criterion,
            supcon_criterion,
            variant,
            supcon_weight=1.0,
            ce_weight=1.0,
            device=device,
            dry_run=dry_run,
        )
        val_metrics = evaluate_model(model, val_loader, device=device, dry_run=dry_run)

        if val_metrics["uar"] > best_val_uar:
            best_val_uar = val_metrics["uar"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience and not dry_run:
                break

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    # Load best checkpoint for zero-shot testing on unseen corpus
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate_model(model, test_loader, device=device, dry_run=dry_run)

    results = {
        "setup_key": setup_key,
        "setup_title": setup_info["title"],
        "variant": variant,
        "variant_title": variant_title,
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
    return results


def run_all_cross_corpus_experiments(dry_run: bool = False, seed: int = 42):
    results_list = []

    logger.info("================================================================================")
    logger.info("STARTING CROSS-CORPUS EVALUATION (15 EXPERIMENTS)")
    logger.info("================================================================================")

    for setup_key, setup_info in CROSS_CORPUS_SETUPS.items():
        logger.info(f"\n--- Setup: {setup_info['title']} ---")
        for variant, variant_title in VARIANTS:
            logger.info(f"Running model: {variant_title} ({variant})...")
            res = run_single_cross_corpus_experiment(
                setup_key=setup_key,
                variant=variant,
                variant_title=variant_title,
                dry_run=dry_run,
                seed=seed,
            )
            results_list.append(res)
            logger.info(
                f" -> Test Acc: {res['test_accuracy']:.4f} | UAR: {res['test_uar']:.4f} | Macro-F1: {res['test_f1_macro']:.4f}"
            )

    # Save summary CSV and JSON
    summary_dir = Path("results/cross_corpus")
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_file = summary_dir / "cross_corpus_summary.csv"
    json_file = summary_dir / "cross_corpus_summary.json"

    fieldnames = ["setup_key", "setup_title", "variant", "variant_title", "test_accuracy", "test_uar", "test_f1_macro", "test_f1_weighted"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_list:
            writer.writerow({k: r[k] for k in fieldnames})

    with open(json_file, "w") as f:
        json.dump(results_list, f, indent=2)

    # Log clean summary table
    logger.info("\n" + "=" * 95)
    logger.info("CROSS-CORPUS EVALUATION RESULTS SUMMARY")
    logger.info("=" * 95)
    header = f"{'Setup':<32} | {'Model':<38} | {'Acc (WAR)':<9} | {'UAR':<9} | {'Macro-F1':<9}"
    logger.info(header)
    logger.info("-" * 95)

    for r in results_list:
        logger.info(
            f"{r['setup_title']:<32} | {r['variant_title']:<38} | {r['test_accuracy']:<9.4f} | {r['test_uar']:<9.4f} | {r['test_f1_macro']:<9.4f}"
        )
    logger.info("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Step 11: Cross-Corpus Evaluation Pipeline")
    parser.add_argument("--all", action="store_true", help="Run all 15 cross-corpus experiment combinations")
    parser.add_argument("--setup", type=str, choices=list(CROSS_CORPUS_SETUPS.keys()), help="Specify cross-corpus setup key")
    parser.add_argument("--variant", type=str, choices=[v[0] for v in VARIANTS], help="Specify model variant")
    parser.add_argument("--dry-run", action="store_true", help="Perform 2-step dry run across experiments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.all or (args.setup is None and args.variant is None):
        run_all_cross_corpus_experiments(dry_run=args.dry_run, seed=args.seed)
    elif args.setup and args.variant:
        variant_title = dict(VARIANTS)[args.variant]
        res = run_single_cross_corpus_experiment(
            setup_key=args.setup,
            variant=args.variant,
            variant_title=variant_title,
            dry_run=args.dry_run,
            seed=args.seed,
        )
        print(json.dumps(res, indent=2))
    else:
        logger.error("Please specify both --setup and --variant, or use --all.")


if __name__ == "__main__":
    main()
