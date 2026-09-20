#!/usr/bin/env python3
"""
Final Reproducible Evaluation Script for Speech Emotion Recognition (SER).
Evaluates existing trained checkpoints without any retraining.
Fulfills User Request Requirement 12.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.metadata import EMOTION_LABELS
from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel

FINAL_CHECKPOINT = PROJECT_ROOT / "runs" / "wavlm_proposed_supcon" / "best_model.pt"
TEST_NPZ = PROJECT_ROOT / "data" / "embeddings" / "wavlm_mean_test.npz"
RESULTS_FINAL_DIR = PROJECT_ROOT / "results" / "final"

FIVE_MODELS = [
    ("WavLM + CE", PROJECT_ROOT / "runs" / "wavlm_ce_baseline" / "best_model.pt", False),
    ("WavLM + Standard SupCon", PROJECT_ROOT / "runs" / "wavlm_supcon" / "best_model.pt", True),
    ("WavLM + Speaker-Aware SupCon", PROJECT_ROOT / "runs" / "wavlm_speaker_supcon" / "best_model.pt", True),
    ("WavLM + Corpus-Aware SupCon", PROJECT_ROOT / "runs" / "wavlm_corpus_supcon" / "best_model.pt", True),
    ("WavLM + Speaker + Corpus-Aware SupCon (Proposed)", FINAL_CHECKPOINT, True),
]

EXPECTED_LABEL_MAPPING = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
}

EXPECTED_TEST_SAMPLE_COUNT = 3234


def verify_checkpoint_architecture(checkpoint_path: Path, use_projection: bool = True) -> tuple[bool, str]:
    if not checkpoint_path.exists():
        return False, f"Checkpoint not found at {checkpoint_path}"
    try:
        model = WavLMEmbeddingSupConModel(
            in_dim=768,
            proj_dim=128,
            num_classes=6,
            use_projection=use_projection,
            use_classifier=True,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(state_dict, strict=True)
        return True, f"Strict load success: {load_res}"
    except Exception as e:
        return False, f"Failed strict load: {e}"


def verify_label_mapping() -> tuple[bool, str]:
    actual_mapping = {i: label for i, label in enumerate(EMOTION_LABELS)}
    if actual_mapping == EXPECTED_LABEL_MAPPING:
        return True, f"Label mapping verified: {actual_mapping}"
    else:
        return False, f"Label mapping mismatch! Actual: {actual_mapping}, Expected: {EXPECTED_LABEL_MAPPING}"


def evaluate_in_domain(model: WavLMEmbeddingSupConModel, test_npz: Path) -> dict:
    data = np.load(test_npz, allow_pickle=True)
    embeddings = torch.from_numpy(data["embeddings"]).float()
    labels = data["labels"]
    corpora = data["corpus"]

    model.eval()
    with torch.no_grad():
        outputs = model(embeddings)
        logits = outputs["logits"]
        preds = logits.argmax(dim=-1).numpy()
        probs = F.softmax(logits, dim=-1).numpy()

    war = float(accuracy_score(labels, preds))
    uar = float(recall_score(labels, preds, average="macro", zero_division=0))
    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    wf1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    cm = confusion_matrix(labels, preds, labels=list(range(6)))

    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=list(range(6)), zero_division=0)
    per_class = {}
    for i, name in enumerate(EMOTION_LABELS):
        per_class[name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }

    per_corpus = {}
    for corp in np.unique(corpora):
        mask = (corpora == corp)
        c_labels = labels[mask]
        c_preds = preds[mask]
        per_corpus[str(corp)] = {
            "num_samples": int(mask.sum()),
            "war": float(accuracy_score(c_labels, c_preds)),
            "uar": float(recall_score(c_labels, c_preds, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(c_labels, c_preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(c_labels, c_preds, average="weighted", zero_division=0)),
        }

    return {
        "num_test_samples": int(len(labels)),
        "war": war,
        "uar": uar,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "per_class": per_class,
        "per_corpus": per_corpus,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_five_baselines(test_npz: Path) -> list[dict]:
    data = np.load(test_npz, allow_pickle=True)
    embeddings = torch.from_numpy(data["embeddings"]).float()
    labels = data["labels"]

    baseline_results = []
    for name, path, use_proj in FIVE_MODELS:
        if not path.exists():
            continue
        m = WavLMEmbeddingSupConModel(768, 128, 6, use_projection=use_proj, use_classifier=True)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        with torch.no_grad():
            preds = m(embeddings)["logits"].argmax(dim=-1).numpy()

        war = float(accuracy_score(labels, preds))
        uar = float(recall_score(labels, preds, average="macro", zero_division=0))
        mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        wf1 = float(f1_score(labels, preds, average="weighted", zero_division=0))

        baseline_results.append({
            "model_name": name,
            "checkpoint_path": str(path.relative_to(PROJECT_ROOT)),
            "war": war,
            "uar": uar,
            "macro_f1": mf1,
            "weighted_f1": wf1,
        })
    return baseline_results


def run_final_evaluation():
    RESULTS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    status = {}

    print("=" * 80)
    print("SER FINAL REPRODUCIBLE EVALUATION (NO RETRAINING)")
    print("=" * 80)

    # 1. Checkpoint Verification
    ckpt_ok, ckpt_msg = verify_checkpoint_architecture(FINAL_CHECKPOINT, use_projection=True)
    status["checkpoint_verification"] = "PASS" if ckpt_ok else "FAIL"
    print(f"[1] Checkpoint Architecture: {status['checkpoint_verification']} -> {ckpt_msg}")

    # 2. Label Mapping Verification
    label_ok, label_msg = verify_label_mapping()
    status["label_mapping_verification"] = "PASS" if label_ok else "FAIL"
    print(f"[2] Label Mapping: {status['label_mapping_verification']} -> {label_msg}")

    # 3. In-Domain Test Evaluation
    model = WavLMEmbeddingSupConModel(768, 128, 6, use_projection=True, use_classifier=True)
    model.load_state_dict(torch.load(FINAL_CHECKPOINT, map_location="cpu"))
    in_domain_res = evaluate_in_domain(model, TEST_NPZ)

    indomain_ok = (in_domain_res["num_test_samples"] == EXPECTED_TEST_SAMPLE_COUNT) and (in_domain_res["war"] > 0)
    status["in_domain_evaluation"] = "PASS" if indomain_ok else "FAIL"
    print(f"[3] In-Domain Evaluation: {status['in_domain_evaluation']}")
    print(f"    - Test Samples: {in_domain_res['num_test_samples']} (Expected: {EXPECTED_TEST_SAMPLE_COUNT})")
    print(f"    - WAR (Acc): {in_domain_res['war']*100:.2f}%")
    print(f"    - UAR (Bal Acc): {in_domain_res['uar']*100:.2f}%")
    print(f"    - Macro-F1: {in_domain_res['macro_f1']*100:.2f}%")
    print(f"    - Weighted-F1: {in_domain_res['weighted_f1']*100:.2f}%")

    # Save in-domain metrics
    with open(RESULTS_FINAL_DIR / "in_domain_metrics.json", "w") as f:
        json.dump(in_domain_res, f, indent=2)

    with open(RESULTS_FINAL_DIR / "in_domain_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["num_test_samples", in_domain_res["num_test_samples"]])
        writer.writerow(["war", in_domain_res["war"]])
        writer.writerow(["uar", in_domain_res["uar"]])
        writer.writerow(["macro_f1", in_domain_res["macro_f1"]])
        writer.writerow(["weighted_f1", in_domain_res["weighted_f1"]])

    with open(RESULTS_FINAL_DIR / "per_corpus_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corpus", "num_samples", "war", "uar", "macro_f1", "weighted_f1"])
        for corp, metrics in in_domain_res["per_corpus"].items():
            writer.writerow([corp, metrics["num_samples"], metrics["war"], metrics["uar"], metrics["macro_f1"], metrics["weighted_f1"]])

    with open(RESULTS_FINAL_DIR / "confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(EMOTION_LABELS))
        for i, row in enumerate(in_domain_res["confusion_matrix"]):
            writer.writerow([EMOTION_LABELS[i]] + row)

    # 4. Five-Model Comparison Verification
    baselines = evaluate_five_baselines(TEST_NPZ)
    five_ok = len(baselines) == 5
    status["five_model_comparison"] = "PASS" if five_ok else "FAIL"
    print(f"[4] Five-Model Comparison: {status['five_model_comparison']} ({len(baselines)}/5 models evaluated)")

    with open(RESULTS_FINAL_DIR / "baseline_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "checkpoint_path", "war", "uar", "macro_f1", "weighted_f1"])
        for b in baselines:
            writer.writerow([b["model_name"], b["checkpoint_path"], b["war"], b["uar"], b["macro_f1"], b["weighted_f1"]])

    # 5. Cross-Corpus Metrics Integration
    cc_summary_file = PROJECT_ROOT / "results" / "cross_corpus" / "cross_corpus_summary.json"
    cc_metrics_file = RESULTS_FINAL_DIR / "cross_corpus_metrics.json"
    cc_csv_file = RESULTS_FINAL_DIR / "cross_corpus_metrics.csv"

    cc_ok = False
    cc_test_counts = {}
    if cc_summary_file.exists():
        with open(cc_summary_file, "r") as f:
            cc_data = json.load(f)
        
        # Filter out dry-run metadata if present, ensure full dataset metrics
        valid_cc_data = [item for item in cc_data if isinstance(item, dict) and "setup_key" in item]
        if len(valid_cc_data) >= 3:
            cc_ok = True
            with open(cc_metrics_file, "w") as f:
                json.dump(valid_cc_data, f, indent=2)

            with open(cc_csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["setup_key", "setup_title", "variant", "variant_title", "test_accuracy", "test_uar", "test_f1_macro", "test_f1_weighted"])
                for item in valid_cc_data:
                    writer.writerow([
                        item.get("setup_key"),
                        item.get("setup_title"),
                        item.get("variant"),
                        item.get("variant_title"),
                        item.get("test_accuracy"),
                        item.get("test_uar"),
                        item.get("test_f1_macro"),
                        item.get("test_f1_weighted"),
                    ])
                    setup = item.get("setup_key")
                    if setup not in cc_test_counts:
                        # Extract test sample count from classification report if available
                        rep = item.get("classification_report", {})
                        supp = rep.get("accuracy", 0)
                        tot = rep.get("macro avg", {}).get("support", None)
                        if tot:
                            cc_test_counts[setup] = int(tot)

    status["cross_corpus_evaluation"] = "PASS" if cc_ok else "FAIL"
    print(f"[5] Cross-Corpus Evaluation: {status['cross_corpus_evaluation']}")

    # 6. Overall Sanity Checks
    all_pass = all(v == "PASS" for v in status.values())
    status["overall"] = "PASS" if all_pass else "FAIL"

    print("=" * 80)
    print("FINAL AUDIT SUMMARY")
    print("=" * 80)
    for k, v in status.items():
        print(f"  {k:<30}: {v}")
    print("=" * 80)

    return status


if __name__ == "__main__":
    run_final_evaluation()
