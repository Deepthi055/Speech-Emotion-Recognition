import os
from math import gcd
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from scipy.signal import resample_poly

EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
}

class SERDataset(Dataset):
    def __init__(self, metadata_path_or_df, target_sr: int = 16000, max_seconds: float = 6.0, max_samples: int = None):
        super().__init__()
        if isinstance(metadata_path_or_df, (str, Path)):
            self.df = pd.read_csv(metadata_path_or_df)
        else:
            self.df = metadata_path_or_df.copy()

        if max_samples is not None and max_samples > 0:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        self.target_sr = target_sr
        self.max_length = int(target_sr * max_seconds) if max_seconds is not None and max_seconds > 0 else None

        self.speakers = sorted(self.df["speaker_id"].unique())
        self.speaker_to_idx = {spk: i for i, spk in enumerate(self.speakers)}

        self.corpora = sorted(self.df["corpus"].unique()) if "corpus" in self.df.columns else ["default"]
        self.corpus_to_idx = {corp: i for i, corp in enumerate(self.corpora)}
        self.labels = np.array([EMOTION_MAP[e] for e in self.df["emotion"]], dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str) -> torch.Tensor:
        path = path.replace("\\", "/")
        if not os.path.isabs(path) and not path.startswith("data/"):
            path = os.path.join("data", path)
        path = str(path).replace("\\", "/")

        waveform, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)

        if sr != self.target_sr:
            g = gcd(sr, self.target_sr)
            waveform = resample_poly(waveform, self.target_sr // g, sr // g).astype(np.float32)

        if self.max_length is not None and len(waveform) > self.max_length:
            waveform = waveform[: self.max_length]

        return torch.from_numpy(waveform)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = row["file_path"] if "file_path" in row else row["audio_path"]
        waveform = self._load_audio(audio_path)

        emotion_str = row["emotion"]
        label = EMOTION_MAP[emotion_str]

        speaker_str = row["speaker_id"]
        speaker_idx = self.speaker_to_idx[speaker_str]

        corpus_str = row.get("corpus", "default")
        corpus_idx = self.corpus_to_idx[corpus_str]

        return {
            "waveform": waveform,
            "label": label,
            "speaker_id": speaker_idx,
            "corpus_id": corpus_idx,
            "speaker_str": speaker_str,
            "corpus_str": corpus_str,
            "emotion_str": emotion_str,
            "file_path": str(audio_path),
        }

def collate_fn(batch):
    waveforms = [item["waveform"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    speaker_ids = torch.tensor([item["speaker_id"] for item in batch], dtype=torch.long)
    corpus_ids = torch.tensor([item["corpus_id"] for item in batch], dtype=torch.long)

    lengths = [len(w) for w in waveforms]
    max_len = max(lengths)

    padded_waveforms = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
    attention_masks = torch.zeros(len(waveforms), max_len, dtype=torch.long)

    for i, w in enumerate(waveforms):
        padded_waveforms[i, : len(w)] = w
        attention_masks[i, : len(w)] = 1

    return {
        "input_values": padded_waveforms,
        "attention_mask": attention_masks,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "corpus_ids": corpus_ids,
        "speaker_strs": [item["speaker_str"] for item in batch],
        "corpus_strs": [item["corpus_str"] for item in batch],
        "file_paths": [item["file_path"] for item in batch],
    }

