import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel
from src.datasets.metadata import EMOTION_LABELS

EMOTION_MAP = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}
INV_MAP = {v: k for k, v in EMOTION_MAP.items()}


def evaluate_proposed_model():
    print("=" * 80)
    print("SER MODEL COMPREHENSIVE EVALUATION & ERROR DIAGNOSIS")
    print("=" * 80)

    # 1. Load Proposed Model
    ckpt_path = PROJECT_ROOT / "runs" / "wavlm_proposed_supcon" / "best_model.pt"
    model = WavLMEmbeddingSupConModel(in_dim=768, proj_dim=128, num_classes=6, use_projection=True, use_classifier=True)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # 2. Load Test Data
    test_npz_path = PROJECT_ROOT / "data" / "embeddings" / "wavlm_mean_test.npz"
    test_npz = np.load(test_npz_path, allow_pickle=True)
    embs = torch.from_numpy(test_npz["embeddings"]).float()
    labels = test_npz["labels"]
    corpora = test_npz["corpus"]
    speakers = test_npz["speaker_ids"]
    file_paths = test_npz["file_paths"]

    with torch.no_grad():
        out = model(embs)
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=-1)

    # 1. Overall Metrics
    acc = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average="macro")
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    cm = confusion_matrix(labels, preds, labels=list(range(6)))

    print("\n--- 1. OVERALL TEST SET METRICS (Proposed Model) ---")
    print(f"Accuracy (WAR)    : {acc*100:.2f}% ({acc:.4f})")
    print(f"UAR (Balanced Acc): {uar*100:.2f}% ({uar:.4f})")
    print(f"Macro-F1          : {f1_macro*100:.2f}% ({f1_macro:.4f})")
    print(f"Weighted-F1       : {f1_weighted*100:.2f}% ({f1_weighted:.4f})")

    print("\n--- 2. COMPLETE CONFUSION MATRIX (Rows=True, Columns=Predicted) ---")
    header_str = f"{'True\\Pred':<10} | " + " | ".join([f"{e:>7}" for e in EMOTION_LABELS])
    print(header_str)
    print("-" * 70)
    for i, row in enumerate(cm):
        row_str = f"{EMOTION_LABELS[i]:<10} | " + " | ".join([f"{v:7d}" for v in row])
        print(row_str)

    print("\n--- 3 & 4. PER-EMOTION METRICS & PREDICTION DISTRIBUTION ---")
    total_samples = len(labels)
    print(f"{'Emotion':<10} | {'True N':<7} | {'Pred N':<7} | {'Pred %':<7} | {'Per-Class Acc':<13} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 90)
    report_dict = classification_report(labels, preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
    for i, emo in enumerate(EMOTION_LABELS):
        true_cnt = int(np.sum(labels == i))
        pred_cnt = int(np.sum(preds == i))
        pred_pct = (pred_cnt / total_samples) * 100
        class_acc = (cm[i, i] / true_cnt * 100) if true_cnt > 0 else 0.0
        prec = report_dict[emo]["precision"] * 100
        rec = report_dict[emo]["recall"] * 100
        f1 = report_dict[emo]["f1-score"] * 100
        print(f"{emo:<10} | {true_cnt:<7d} | {pred_cnt:<7d} | {pred_pct:<6.2f}% | {class_acc:<12.2f}% | {prec:<8.2f}% | {rec:<7.2f}% | {f1:<7.2f}%")

    # 5. Baseline Comparisons
    print("\n--- 5. COMPARISON AGAINST ALL IN-DOMAIN BASELINES ---")
    summary_file = PROJECT_ROOT / "runs" / "comparison_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        print(f"{'Model Architecture':<45} | {'Test Acc':<10} | {'Test UAR':<10} | {'Macro-F1':<10}")
        print("-" * 82)
        for item in summary:
            print(f"{item['method']:<45} | {item['test_accuracy']*100:8.2f}% | {item['test_uar']*100:8.2f}% | {item['test_f1_macro']*100:8.2f}%")

    # 6. Errors separately by Corpus
    print("\n--- 6. PER-CORPUS METRICS & CONFUSION MATRICES ---")
    for corp in sorted(list(np.unique(corpora))):
        mask = (corpora == corp)
        c_labels = labels[mask]
        c_preds = preds[mask]
        c_acc = accuracy_score(c_labels, c_preds)
        c_uar = recall_score(c_labels, c_preds, average="macro", zero_division=0)
        c_f1 = f1_score(c_labels, c_preds, average="macro", zero_division=0)
        print(f"\nCorpus: {corp} (N={len(c_labels)}) -> Acc: {c_acc*100:.2f}% | UAR: {c_uar*100:.2f}% | Macro-F1: {c_f1*100:.2f}%")
        c_cm = confusion_matrix(c_labels, c_preds, labels=list(range(6)))
        print(f"  {header_str}")
        for i, row in enumerate(c_cm):
            row_str = f"  {EMOTION_LABELS[i]:<10} | " + " | ".join([f"{v:7d}" for v in row])
            print(row_str)

    # 7. Errors by Cross-Corpus Test Condition
    print("\n--- 7. CROSS-CORPUS ZERO-SHOT BENCHMARKS ---")
    cc_summary_file = PROJECT_ROOT / "results" / "cross_corpus" / "cross_corpus_summary.json"
    if cc_summary_file.exists():
        with open(cc_summary_file, "r") as f:
            cc_data = json.load(f)
        print(f"{'Setup Title':<32} | {'Model Variant':<38} | {'Test Acc':<9} | {'UAR':<9} | {'Macro-F1':<9}")
        print("-" * 97)
        for row in cc_data:
            print(f"{row['setup_title']:<32} | {row['variant_title']:<38} | {row['test_accuracy']*100:7.2f}% | {row['test_uar']*100:7.2f}% | {row['test_f1_macro']*100:7.2f}%")

    # 8. 10 Representative Misclassified Samples
    print("\n--- 8. 10 REPRESENTATIVE MISCLASSIFIED SAMPLES ---")
    misclassified_idx = np.where(labels != preds)[0]
    np.random.seed(42)
    sample_indices = np.random.choice(misclassified_idx, size=min(10, len(misclassified_idx)), replace=False)

    for rank, idx in enumerate(sample_indices, 1):
        fpath = Path(file_paths[idx]).name
        corp = corpora[idx]
        spk = speakers[idx]
        t_emo = EMOTION_LABELS[labels[idx]]
        p_emo = EMOTION_LABELS[preds[idx]]
        conf = probs[idx][preds[idx]]
        prob_dist = [round(float(p), 4) for p in probs[idx]]

        print(f"Sample #{rank:02d}:")
        print(f"  Filename        : {fpath}")
        print(f"  Corpus / Speaker: {corp} / {spk}")
        print(f"  True Emotion    : {t_emo.upper()} (idx {labels[idx]})")
        print(f"  Predicted       : {p_emo.upper()} (idx {preds[idx]}) [Confidence: {conf*100:.2f}%]")
        print(f"  Probability Dist: {prob_dist}")
        print(f"                    (angry: {prob_dist[0]}, disgust: {prob_dist[1]}, fear: {prob_dist[2]}, happy: {prob_dist[3]}, neutral: {prob_dist[4]}, sad: {prob_dist[5]})")
        print("-" * 75)

    # 11 & 12. Inspect Training Config & Representation
    print("\n--- 11 & 12. TRAINING CONFIGURATION & ARCHITECTURE AUDIT ---")
    cfg_file = PROJECT_ROOT / "configs" / "wavlm_proposed_supcon.yaml"
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            print(f.read())


if __name__ == "__main__":
    evaluate_proposed_model()
