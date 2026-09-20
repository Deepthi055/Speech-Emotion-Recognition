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
        if self.speaker_ids is not None:
            _, spk_indices = np.unique(self.speaker_ids, return_inverse=True)
            self.speaker_encoded = spk_indices
        else:
            self.speaker_encoded = np.zeros(len(self.labels), dtype=int)

        if self.corpus is not None:
            _, corpus_indices = np.unique(self.corpus, return_inverse=True)
            self.corpus_encoded = corpus_indices
        else:
            self.corpus_encoded = np.zeros(len(self.labels), dtype=int)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        emb = torch.from_numpy(self.embeddings[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        spk_id = torch.tensor(self.speaker_encoded[idx], dtype=torch.long)
        corpus_id = torch.tensor(self.corpus_encoded[idx], dtype=torch.long)
        return {
            "embedding": emb,
            "label": label,
            "speaker_id": spk_id,
            "corpus_id": corpus_id,
        }

