import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel
from src.datasets.metadata import EMOTION_LABELS

EMOTION_MAP = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}


def evaluate_new_experiment():
    print("=" * 80)
    print("POINT 8: EVALUATION OF NEW LOSS-WEIGHTED EXPERIMENT")
    print(" (CE weight = 1.5, SupCon weight = 0.5)")
    print("=" * 80)

    # Load New Model
    new_ckpt_path = PROJECT_ROOT / "runs" / "wavlm_lossweight_experiment" / "best_model.pt"
    model_new = WavLMEmbeddingSupConModel(in_dim=768, proj_dim=128, num_classes=6, use_projection=True, use_classifier=True)
    model_new.load_state_dict(torch.load(new_ckpt_path, map_location="cpu"), strict=True)
    model_new.eval()

    # Load Baseline Proposed Model
    orig_ckpt_path = PROJECT_ROOT / "runs" / "wavlm_proposed_supcon" / "best_model.pt"
    model_orig = WavLMEmbeddingSupConModel(in_dim=768, proj_dim=128, num_classes=6, use_projection=True, use_classifier=True)
    model_orig.load_state_dict(torch.load(orig_ckpt_path, map_location="cpu"), strict=True)
    model_orig.eval()

    # Load Test Data
    test_npz_path = PROJECT_ROOT / "data" / "embeddings" / "wavlm_mean_test.npz"
    test_npz = np.load(test_npz_path, allow_pickle=True)
    embs = torch.from_numpy(test_npz["embeddings"]).float()
    labels = test_npz["labels"]
    corpora = test_npz["corpus"]

    with torch.no_grad():
        preds_new = model_new(embs)["logits"].argmax(dim=-1).numpy()
        preds_orig = model_orig(embs)["logits"].argmax(dim=-1).numpy()

    # Metrics
    war_new = accuracy_score(labels, preds_new)
    uar_new = recall_score(labels, preds_new, average="macro", zero_division=0)
    f1_new = f1_score(labels, preds_new, average="macro", zero_division=0)
    f1_w_new = f1_score(labels, preds_new, average="weighted", zero_division=0)

    war_orig = accuracy_score(labels, preds_orig)
    uar_orig = recall_score(labels, preds_orig, average="macro", zero_division=0)
    f1_orig = f1_score(labels, preds_orig, average="macro", zero_division=0)

    print("\n--- OVERALL TEST METRICS COMPARISON ---")
    print(f"{'Metric':<20} | {'Original (1.0/1.0)':<20} | {'New LossWeight (1.5/0.5)':<25} | {'Diff':<10}")
    print("-" * 80)
    print(f"{'Accuracy (WAR)':<20} | {war_orig*100:19.2f}% | {war_new*100:24.2f}% | {(war_new-war_orig)*100:+9.2f}%")
    print(f"{'UAR (Bal Acc)':<20} | {uar_orig*100:19.2f}% | {uar_new*100:24.2f}% | {(uar_new-uar_orig)*100:+9.2f}%")
    print(f"{'Macro-F1':<20} | {f1_orig*100:19.2f}% | {f1_new*100:24.2f}% | {(f1_new-f1_orig)*100:+9.2f}%")

    # Confusion Matrix New Model
    cm_new = confusion_matrix(labels, preds_new, labels=list(range(6)))
    cm_orig = confusion_matrix(labels, preds_orig, labels=list(range(6)))

    print("\n--- NEW EXPERIMENT CONFUSION MATRIX ---")
    header_str = f"{'True\\Pred':<10} | " + " | ".join([f"{e:>7}" for e in EMOTION_LABELS])
    print(header_str)
    print("-" * 70)
    for i, row in enumerate(cm_new):
        row_str = f"{EMOTION_LABELS[i]:<10} | " + " | ".join([f"{v:7d}" for v in row])
        print(row_str)

    # Per class metrics breakdown
    rep_new = classification_report(labels, preds_new, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
    rep_orig = classification_report(labels, preds_orig, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)

    print("\n--- PER-CLASS RECALL & F1 COMPARISON (Focus on Neutral & Sad) ---")
    print(f"{'Emotion':<10} | {'Orig Recall':<12} | {'New Recall':<12} | {'Recall Diff':<12} | {'Orig F1':<10} | {'New F1':<10}")
    print("-" * 80)
    for emo in EMOTION_LABELS:
        r_o = rep_orig[emo]["recall"] * 100
        r_n = rep_new[emo]["recall"] * 100
        f_o = rep_orig[emo]["f1-score"] * 100
        f_n = rep_new[emo]["f1-score"] * 100
        print(f"{emo:<10} | {r_o:11.2f}% | {r_n:11.2f}% | {(r_n-r_o):+11.2f}% | {f_o:9.2f}% | {f_n:9.2f}%")

    # Per corpus metrics
    print("\n--- PER-CORPUS METRICS (NEW EXPERIMENT) ---")
    for cname in ["CREMA-D", "RAVDESS", "IEMOCAP"]:
        cmask = (corpora == cname)
        c_acc_n = accuracy_score(labels[cmask], preds_new[cmask])
        c_uar_n = recall_score(labels[cmask], preds_new[cmask], average="macro", zero_division=0)
        c_f1_n = f1_score(labels[cmask], preds_new[cmask], average="macro", zero_division=0)

        c_acc_o = accuracy_score(labels[cmask], preds_orig[cmask])
        print(f"Corpus: {cname:<8} -> New Acc: {c_acc_n*100:5.2f}% (Orig: {c_acc_o*100:5.2f}%) | New UAR: {c_uar_n*100:5.2f}% | New Macro-F1: {c_f1_n*100:5.2f}%")


if __name__ == "__main__":
    evaluate_new_experiment()
