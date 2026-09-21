#!/usr/bin/env python3
"""
Test script for verifying speech emotion recognition model predictions on audio files from the dataset.
"""

import json
import urllib.request
import io
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.metadata import EMOTION_LABELS
from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel
from transformers import WavLMModel

def test_model_direct(csv_path: Path, checkpoint_path: Path, num_samples_per_emotion: int = 3):
    print("================================================================")
    print("DIRECT IN-PYTHON MODEL INFERENCE TEST ON RAW AUDIO FILES")
    print("================================================================")

    df = pd.read_csv(csv_path)

    samples = []
    for emotion in EMOTION_LABELS:
        sub = df[df['emotion'] == emotion]
        if len(sub) > 0:
            samples.append(sub.sample(min(num_samples_per_emotion, len(sub)), random_state=42))

    sample_df = pd.concat(samples)

    print("Loading WavLM backbone (microsoft/wavlm-base)...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
    wavlm.eval()

    print(f"Loading trained classifier ({checkpoint_path.name})...")
    classifier = WavLMEmbeddingSupConModel(
        in_dim=768,
        proj_dim=128,
        num_classes=6,
        use_projection=True,
        use_classifier=True,
    )
    classifier.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    classifier.eval()

    correct = 0
    total = 0

    print("\nStarting per-file evaluation:\n")
    for _, row in sample_df.iterrows():
        fpath = row['file_path']
        gt = row['emotion']
        corpus = row['corpus']

        # Read audio & resample to 16kHz
        wav, sr = sf.read(fpath, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != 16000:
            g = gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)

        inp = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            hidden = wavlm(inp).last_hidden_state
            emb = hidden.mean(dim=1)
            logits = classifier(emb)["logits"]
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        pred_idx = int(np.argmax(probs))
        pred_emotion = EMOTION_LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        is_correct = (pred_emotion == gt)
        if is_correct:
            correct += 1
        total += 1

        status = "✓ PASS" if is_correct else "✗ FAIL"
        filename = Path(fpath).name
        print(f"[{status}] Corpus: {corpus:<8} | GT: {gt:<8} | Pred: {pred_emotion:<8} (Conf: {confidence:.2%}) | File: {filename}")

    acc = (correct / total) * 100
    print(f"\nDirect Python Model Accuracy on {total} sample files: {correct}/{total} ({acc:.2f}%)")
    return acc

def main():
    test_csv = PROJECT_ROOT / "data" / "metadata" / "unified_test.csv"
    checkpoint = PROJECT_ROOT / "runs" / "wavlm_proposed_supcon" / "best_model.pt"

    if not test_csv.exists():
        print(f"Error: {test_csv} not found.")
        return

    test_model_direct(test_csv, checkpoint)

if __name__ == "__main__":
    main()
