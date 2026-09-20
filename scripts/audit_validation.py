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

from src.datasets.metadata import EMOTION_LABELS
from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel

EMOTION_MAP = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}


def audit_splits_and_metadata():
    print("=" * 80)
    print("1. VERIFYING TEST & SPLIT INTEGRITY")
    print("=" * 80)

    meta_dir = PROJECT_ROOT / "data" / "metadata"
    df_train = pd.read_csv(meta_dir / "unified_train.csv")
    df_val = pd.read_csv(meta_dir / "unified_val.csv")
    df_test = pd.read_csv(meta_dir / "unified_test.csv")

    print(f"Total rows -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Speaker overlap check
    spk_train = set(df_train["speaker_id"])
    spk_val = set(df_val["speaker_id"])
    spk_test = set(df_test["speaker_id"])

    overlap_tr_val = spk_train.intersection(spk_val)
    overlap_tr_te = spk_train.intersection(spk_test)
    overlap_val_te = spk_val.intersection(spk_test)

    print(f"\nSpeaker Count -> Train: {len(spk_train)}, Val: {len(spk_val)}, Test: {len(spk_test)}")
    print(f"Speaker Overlap (Train - Val): {len(overlap_tr_val)} {overlap_tr_val}")
    print(f"Speaker Overlap (Train - Test): {len(overlap_tr_te)} {overlap_tr_te}")
    print(f"Speaker Overlap (Val - Test): {len(overlap_val_te)} {overlap_val_te}")

    # File path / Utterance overlap check
    file_train = set(df_train["file_path"])
    file_val = set(df_val["file_path"])
    file_test = set(df_test["file_path"])

    f_overlap_tr_val = file_train.intersection(file_val)
    f_overlap_tr_te = file_train.intersection(file_test)
    f_overlap_val_te = file_val.intersection(file_test)

    print(f"\nUtterance File Overlap (Train-Val): {len(f_overlap_tr_val)}")
    print(f"Utterance File Overlap (Train-Test): {len(f_overlap_tr_te)}")
    print(f"Utterance File Overlap (Val-Test): {len(f_overlap_val_te)}")

    is_speaker_independent = (len(overlap_tr_val) == 0 and len(overlap_tr_te) == 0 and len(overlap_val_te) == 0)
    print(f"\nIs split Genuinely Speaker-Independent? {'YES [VERIFIED]' if is_speaker_independent else 'NO [FLAW DETECTED]'}")

    # NPZ files check vs CSV
    npz_dir = PROJECT_ROOT / "data" / "embeddings"
    for split, df_split in zip(["train", "val", "test"], [df_train, df_val, df_test]):
        npz_data = np.load(npz_dir / f"wavlm_mean_{split}.npz", allow_pickle=True)
        embs = npz_data["embeddings"]
        labels = npz_data["labels"]
        print(f"NPZ {split:<5} -> shape: {embs.shape}, labels: {labels.shape}, matches CSV length: {len(embs) == len(df_split)}")


def audit_label_distribution():
    print("\n" + "=" * 80)
    print("2. COMPLETE LABEL DISTRIBUTION TABLE ACROSS SPLITS")
    print("=" * 80)

    meta_dir = PROJECT_ROOT / "data" / "metadata"
    df_train = pd.read_csv(meta_dir / "unified_train.csv")
    df_val = pd.read_csv(meta_dir / "unified_val.csv")
    df_test = pd.read_csv(meta_dir / "unified_test.csv")

    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)

    print(f"{'Corpus':<10} | {'Emotion':<10} | {'Train Count':<12} | {'Val Count':<12} | {'Test Count':<12} | {'Total':<10}")
    print("-" * 80)

    for corp in sorted(df_full["corpus"].unique()):
        for emo in EMOTION_LABELS:
            tr_c = len(df_train[(df_train["corpus"] == corp) & (df_train["emotion"] == emo)])
            va_c = len(df_val[(df_val["corpus"] == corp) & (df_val["emotion"] == emo)])
            te_c = len(df_test[(df_test["corpus"] == corp) & (df_test["emotion"] == emo)])
            tot = tr_c + va_c + te_c
            print(f"{corp:<10} | {emo:<10} | {tr_c:<12d} | {va_c:<12d} | {te_c:<12d} | {tot:<10d}")
        print("-" * 80)


def audit_cross_corpus_experiments():
    print("\n" + "=" * 80)
    print("3. CROSS-CORPUS EXPERIMENTS AUDIT")
    print("=" * 80)

    cc_dir = PROJECT_ROOT / "results" / "cross_corpus"
    setups = ["crema_ravdess_to_iemocap", "crema_iemocap_to_ravdess", "ravdess_iemocap_to_cremad"]

    for setup_key in setups:
        mfile = cc_dir / setup_key / "proposed_supcon" / "metrics.json"
        if not mfile.exists():
            print(f"Missing cross-corpus metrics for {setup_key}")
            continue

        with open(mfile, "r") as f:
            res = json.load(f)

        print(f"\n=== Cross-Corpus Setup: {res['setup_title']} ===")
        print(f"Variant              : {res['variant_title']}")
        print(f"Test Accuracy (WAR)  : {res['test_accuracy']*100:.2f}%")
        print(f"Test UAR             : {res['test_uar']*100:.2f}%")
        print(f"Test Macro-F1        : {res['test_f1_macro']*100:.2f}%")

        cm = np.array(res["confusion_matrix"])
        print("\nConfusion Matrix:")
        header_str = f"{'True\\Pred':<10} | " + " | ".join([f"{e:>7}" for e in EMOTION_LABELS])
        print(header_str)
        for i, row in enumerate(cm):
            row_str = f"{EMOTION_LABELS[i]:<10} | " + " | ".join([f"{v:7d}" for v in row])
            print(row_str)

        rep = res["classification_report"]
        print("\nPer-Class Breakdown:")
        print(f"{'Emotion':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
        for emo in EMOTION_LABELS:
            if emo in rep:
                print(f"{emo:<10} | {rep[emo]['precision']*100:9.2f}% | {rep[emo]['recall']*100:9.2f}% | {rep[emo]['f1-score']*100:9.2f}% | {int(rep[emo]['support']):<10d}")


if __name__ == "__main__":
    audit_splits_and_metadata()
    audit_label_distribution()
    audit_cross_corpus_experiments()
