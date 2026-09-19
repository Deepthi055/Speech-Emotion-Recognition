from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, npz_path: str | Path, max_samples: int = None):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.embeddings = data["embeddings"]  # (N, 768)
        self.labels = data["labels"]          # (N,)
        self.speaker_ids = data["speaker_ids"] if "speaker_ids" in data else None
        self.corpus = data["corpus"] if "corpus" in data else None
        self.file_paths = data["file_paths"] if "file_paths" in data else None

        if max_samples is not None and max_samples > 0:
            self.embeddings = self.embeddings[:max_samples]
            self.labels = self.labels[:max_samples]
            if self.speaker_ids is not None:
                self.speaker_ids = self.speaker_ids[:max_samples]
            if self.corpus is not None:
                self.corpus = self.corpus[:max_samples]
            if self.file_paths is not None:
                self.file_paths = self.file_paths[:max_samples]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        emb = torch.from_numpy(self.embeddings[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "embedding": emb,
            "label": label,
        }
