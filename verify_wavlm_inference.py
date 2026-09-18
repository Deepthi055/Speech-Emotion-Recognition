"""
verify_wavlm_inference.py
─────────────────────────
Pipeline verification: load 5-10 CREMA-D samples, resample to 16 kHz,
run WavLM Base (microsoft/wavlm-base) forward pass, apply mean pooling
over the time dimension, and print both the raw hidden-state shape and
the final 768-D sentence embedding shape.

Usage:
    python verify_wavlm_inference.py
"""

import sys
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WavLMModel, AutoFeatureExtractor


MODEL_ID      = "microsoft/wavlm-base"
TARGET_SR     = 16_000          # WavLM expects 16 kHz mono
N_SAMPLES     = 10              # how many files to run (capped by availability)
AUDIO_DIR     = Path("data/raw/CREMA-D/AudioWAV")

def load_and_resample(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load a WAV file with soundfile and return a 1-D mono waveform at *target_sr*."""
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)

    # Mix down to mono
    waveform = waveform.mean(axis=1)  # (T,)

    # Resample if necessary using polyphase filter (high quality, no extra codec)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        waveform = resample_poly(waveform, target_sr // g, sr // g).astype(np.float32)

    return torch.from_numpy(waveform)  # shape: (T,)


def mean_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """Average-pool over the time dimension.

    Args:
        hidden_states: shape (batch, time, hidden_size)

    Returns:
        shape (batch, hidden_size)
    """
    return hidden_states.mean(dim=1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}")

    audio_dir = Path(__file__).parent / AUDIO_DIR
    if not audio_dir.exists():
        sys.exit(
            f"[ERROR] Audio directory not found: {audio_dir}\n"
            "Make sure CREMA-D is placed under data/raw/CREMA-D/AudioWAV/"
        )

    wav_files = sorted(audio_dir.glob("*.wav"))[:N_SAMPLES]
    if not wav_files:
        sys.exit(f"[ERROR] No .wav files found in {audio_dir}")

    print(f"\nFound {len(wav_files)} audio file(s) to process.")

    print(f"\nLoading model  : {MODEL_ID}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model             = WavLMModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print("Model loaded successfully.\n")
    print("=" * 70)

    with torch.no_grad():
        for idx, wav_path in enumerate(wav_files, start=1):
            # Load & resample
            waveform = load_and_resample(wav_path, TARGET_SR)  # (T,)
            duration = waveform.shape[0] / TARGET_SR

            # Feature extraction (normalisation + padding)
            inputs = feature_extractor(
                waveform.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(device)   # (1, T)

            # Forward pass
            outputs       = model(input_values)
            hidden_states = outputs.last_hidden_state           # (1, time_frames, 768)

            # Mean pooling
            embedding = mean_pool(hidden_states)                # (1, 768)

            print(
                f"[{idx:02d}] {wav_path.name}\n"
                f"     Duration       : {duration:.2f} s  |  "
                f"Samples @ {TARGET_SR} Hz : {waveform.shape[0]:,}\n"
                f"     Hidden-state   : {tuple(hidden_states.shape)}  "
                f"(batch, time_frames, hidden_size)\n"
                f"     Embedding      : {tuple(embedding.shape)}  "
                f"(batch, hidden_size)\n"
            )

    print("=" * 70)
    print("Pipeline verification complete -- WavLM Base inference OK.")


if __name__ == "__main__":
    main()
