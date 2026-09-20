import sys
import argparse
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
from transformers import WavLMModel

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel
from src.datasets.metadata import EMOTION_LABELS

MODEL_ID = "microsoft/wavlm-base"
TARGET_SR = 16000
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "wavlm_proposed_supcon" / "best_model.pt"


def load_and_preprocess_audio(audio_path: str | Path, target_sr: int = TARGET_SR):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, orig_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)

    # Convert multi-channel to mono by averaging channels
    if waveform.ndim > 1 and waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform.squeeze(axis=-1)

    if waveform.size == 0:
        raise ValueError(f"Audio file {audio_path} is empty.")

    # Resample to 16,000 Hz if necessary
    if orig_sr != target_sr:
        g = gcd(orig_sr, target_sr)
        waveform = resample_poly(waveform, target_sr // g, orig_sr // g).astype(np.float32)

    return torch.from_numpy(waveform), orig_sr


def run_standalone_inference(audio_path: str | Path, checkpoint_path: str | Path = DEFAULT_CHECKPOINT):
    audio_path = Path(audio_path)
    checkpoint_path = Path(checkpoint_path)

    print("==================================================")
    print("STANDALONE SPEECH EMOTION RECOGNITION INFERENCE")
    print("==================================================")
    print(f"Input file           : {audio_path.resolve()}")
    print(f"Model Checkpoint     : {checkpoint_path.resolve()}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Load Audio
    waveform, orig_sr = load_and_preprocess_audio(audio_path, TARGET_SR)
    print(f"Detected Sample Rate : {orig_sr} Hz")
    print(f"Target Sample Rate   : {TARGET_SR} Hz")
    print(f"Waveform Tensor Shape: {tuple(waveform.shape)}")

    # 2. Extract WavLM Embedding
    device = torch.device("cpu")
    print(f"\nLoading WavLM backbone ({MODEL_ID})...")
    wavlm = WavLMModel.from_pretrained(MODEL_ID).to(device)
    wavlm.eval()
    for p in wavlm.parameters():
        p.requires_grad = False

    input_values = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = wavlm(input_values)
        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1)  # (1, 768)

    print(f"WavLM Embedding Shape: {tuple(embedding.shape)}")

    # 3. Load Classifier Model & Checkpoint
    classifier = WavLMEmbeddingSupConModel(
        in_dim=768,
        proj_dim=128,
        num_classes=len(EMOTION_LABELS),
        use_projection=True,
        use_classifier=True,
    ).to(device)

    print("\nCheckpoint Loading (Strict Mode):")
    state_dict = torch.load(checkpoint_path, map_location=device)
    load_res = classifier.load_state_dict(state_dict, strict=True)
    print(f"  Status             : Strict state_dict matched cleanly ({load_res})")

    param_count = sum(p.numel() for p in classifier.parameters())
    trainable_count = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"  Total Parameters   : {param_count:,}")
    print(f"  Trainable Params   : {trainable_count:,}")

    # 4. Label Mapping
    print("\nLabel-to-Index Mapping (Training & Inference):")
    for idx, label in enumerate(EMOTION_LABELS):
        print(f"  {idx} -> {label}")

    # 5. Run Classifier Inference
    classifier.eval()
    with torch.no_grad():
        model_out = classifier(embedding)
        logits = model_out["logits"].squeeze(0)  # (6,)
        probs = F.softmax(logits, dim=-1)

    logits_np = logits.cpu().numpy()
    probs_np = probs.cpu().numpy()
    pred_idx = int(torch.argmax(probs).item())
    pred_emotion = EMOTION_LABELS[pred_idx]

    print("\nInference Results:")
    print(f"  Raw Logits         : {np.round(logits_np, 4).tolist()}")
    print("  Emotion Softmax Probabilities:")
    for idx, label in enumerate(EMOTION_LABELS):
        marker = " <--- PREDICTED" if idx == pred_idx else ""
        print(f"    [{idx}] {label:<8}: {probs_np[idx]*100:6.2f}% ({probs_np[idx]:.4f}){marker}")

    print("\n==================================================")
    print(f"FINAL PREDICTED EMOTION: {pred_emotion.upper()} (Confidence: {probs_np[pred_idx]*100:.2f}%)")
    print("==================================================")

    return {
        "audio_path": str(audio_path),
        "orig_sr": orig_sr,
        "waveform_shape": list(waveform.shape),
        "embedding_shape": list(embedding.shape),
        "checkpoint": str(checkpoint_path),
        "label_mapping": {idx: label for idx, label in enumerate(EMOTION_LABELS)},
        "logits": logits_np.tolist(),
        "probabilities": {label: float(probs_np[i]) for i, label in enumerate(EMOTION_LABELS)},
        "predicted_emotion": pred_emotion,
        "confidence": float(probs_np[pred_idx]),
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone SER Inference Verification Script")
    parser.add_argument("audio_path", type=str, help="Path to input audio file (.wav)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to model checkpoint")
    args = parser.parse_args()

    run_standalone_inference(args.audio_path, args.checkpoint)


if __name__ == "__main__":
    main()
