import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel
from src.models.baseline import WavLMClassifier
from src.datasets.metadata import EMOTION_LABELS


def compare_all_baselines():
    test_npz = np.load(PROJECT_ROOT / "data" / "embeddings" / "wavlm_mean_test.npz", allow_pickle=True)
    embs = torch.from_numpy(test_npz["embeddings"]).float()
    labels = test_npz["labels"]
    corpora = test_npz["corpus"]

    models_info = [
        ("WavLM + CE (Baseline)", "runs/wavlm_ce_baseline/best_model.pt", "ce"),
        ("Standard SupCon", "runs/wavlm_supcon/best_model.pt", "supcon"),
        ("Speaker-Aware SupCon", "runs/wavlm_speaker_supcon/best_model.pt", "supcon"),
        ("Corpus-Aware SupCon", "runs/wavlm_corpus_supcon/best_model.pt", "supcon"),
        ("Speaker + Corpus-Aware SupCon (Proposed)", "runs/wavlm_proposed_supcon/best_model.pt", "supcon"),
    ]

    print("=" * 105)
    print("POINT 6: COMPARISON TABLE ACROSS ALL 5 VARIANTS")
    print("=" * 105)
    header = f"{'Model Variant':<40} | {'WAR':<8} | {'UAR':<8} | {'Macro-F1':<8} | {'CREMA-D Acc':<11} | {'RAVDESS Acc':<11} | {'IEMOCAP Acc':<11}"
    print(header)
    print("-" * 105)

    results_table = []
    for name, ckpt_rel, mtype in models_info:
        ckpt = PROJECT_ROOT / ckpt_rel
        if not ckpt.exists():
            print(f"Skipping {name}: {ckpt} not found")
            continue

        if mtype == "ce":
            model = WavLMClassifier(in_dim=768, num_classes=6)
        else:
            model = WavLMEmbeddingSupConModel(in_dim=768, proj_dim=128, num_classes=6, use_projection=True, use_classifier=True)

        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        model.eval()

        with torch.no_grad():
            if mtype == "ce":
                logits = model(embs)
            else:
                logits = model(embs)["logits"]
            preds = logits.argmax(dim=-1).numpy()

        war = accuracy_score(labels, preds)
        uar = recall_score(labels, preds, average="macro", zero_division=0)
        f1_m = f1_score(labels, preds, average="macro", zero_division=0)

        # Per corpus accuracy
        c_accs = {}
        for cname in ["CREMA-D", "RAVDESS", "IEMOCAP"]:
            cmask = (corpora == cname)
            c_accs[cname] = accuracy_score(labels[cmask], preds[cmask])

        results_table.append({
            "name": name, "war": war, "uar": uar, "f1_m": f1_m,
            "cremad": c_accs["CREMA-D"], "ravdess": c_accs["RAVDESS"], "iemocap": c_accs["IEMOCAP"]
        })

        c_c = c_accs["CREMA-D"] * 100
        c_r = c_accs["RAVDESS"] * 100
        c_i = c_accs["IEMOCAP"] * 100
        print(f"{name:<40} | {war*100:6.2f}% | {uar*100:6.2f}% | {f1_m*100:6.2f}% | {c_c:9.2f}% | {c_r:9.2f}% | {c_i:9.2f}%")

    print("\n--- Absolute Improvement over WavLM + CE Baseline ---")
    ce_war = results_table[0]["war"]
    ce_uar = results_table[0]["uar"]
    ce_f1 = results_table[0]["f1_m"]

    for row in results_table[1:]:
        d_war = (row["war"] - ce_war) * 100
        d_uar = (row["uar"] - ce_uar) * 100
        d_f1 = (row["f1_m"] - ce_f1) * 100
        print(f"{row['name']:<40} -> WAR: {d_war:+5.2f}% | UAR: {d_uar:+5.2f}% | Macro-F1: {d_f1:+5.2f}%")

    print("\n--- Proposed Model Improvement over Standard SupCon ---")
    std_sup_war = results_table[1]["war"]
    std_sup_uar = results_table[1]["uar"]
    std_sup_f1 = results_table[1]["f1_m"]

    prop_war = results_table[4]["war"]
    prop_uar = results_table[4]["uar"]
    prop_f1 = results_table[4]["f1_m"]

    print(f"Proposed vs Standard SupCon -> WAR: {(prop_war-std_sup_war)*100:+5.2f}% | UAR: {(prop_uar-std_sup_uar)*100:+5.2f}% | Macro-F1: {(prop_f1-std_sup_f1)*100:+5.2f}%")


if __name__ == "__main__":
    compare_all_baselines()
