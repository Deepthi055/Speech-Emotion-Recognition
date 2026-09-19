import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from math import gcd
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import WavLMModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = "microsoft/wavlm-base"
TARGET_SR = 16000
EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
}

def load_and_resample(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)

    if sr != target_sr:
        g = gcd(sr, target_sr)
        waveform = resample_poly(waveform, target_sr // g, sr // g).astype(np.float32)

    return torch.from_numpy(waveform)

@torch.no_grad()
def extract_embeddings_for_split(
    df: pd.DataFrame,
    model: WavLMModel,
    output_path: Path,
    device: torch.device,
    max_samples: int = None,
):
    model.eval()
    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].reset_index(drop=True)

    embeddings_list = []
    labels_list = []
    speaker_ids_list = []
    corpus_list = []
    file_paths_list = []

    print(f"Extracting embeddings for {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {output_path.stem}"):
        fpath = row["file_path"]
        waveform = load_and_resample(fpath, TARGET_SR)  # (T,)
        input_values = waveform.unsqueeze(0).to(device)  # (1, T)

        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state  # (1, frames, 768)

        # Mean pooling over time frames
        mean_pooled = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)

        embeddings_list.append(mean_pooled)
        labels_list.append(EMOTION_MAP[row["emotion"]])
        speaker_ids_list.append(row["speaker_id"])
        corpus_list.append(row["corpus"])
        file_paths_list.append(fpath)

    embeddings_arr = np.array(embeddings_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)
    speaker_ids_arr = np.array(speaker_ids_list)
    corpus_arr = np.array(corpus_list)
    file_paths_arr = np.array(file_paths_list)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings_arr,
        labels=labels_arr,
        speaker_ids=speaker_ids_arr,
        corpus=corpus_arr,
        file_paths=file_paths_arr,
    )
    print(f"Saved cached embeddings to {output_path} with shape {embeddings_arr.shape}")

def main():
    parser = argparse.ArgumentParser(description="Extract and cache frozen WavLM embeddings (CPU-only)")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata", help="Directory containing unified CSVs")
    parser.add_argument("--output_dir", type=str, default="data/embeddings", help="Directory to save .npz files")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap for testing")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading WavLM backbone ({MODEL_ID}) on device: {device}...")
    model = WavLMModel.from_pretrained(MODEL_ID).to(device)

    # Freeze all parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)

    for split in ["train", "val", "test"]:
        csv_file = metadata_dir / f"unified_{split}.csv"
        if not csv_file.exists():
            print(f"Metadata file {csv_file} not found. Skipping.")
            continue

        df = pd.read_csv(csv_file)
        out_npz = output_dir / f"wavlm_mean_{split}.npz"
        extract_embeddings_for_split(df, model, out_npz, device, max_samples=args.max_samples)

    print("Embedding extraction completed.")

if __name__ == "__main__":
    main()
