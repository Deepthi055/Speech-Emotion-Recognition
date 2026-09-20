from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path | list[str | Path],
        include_corpora: list[str] | set[str] = None,
        max_samples: int = None,
    ):
        super().__init__()
        if isinstance(npz_path, (list, tuple)):
            npz_paths = [Path(p) for p in npz_path]
        else:
            npz_paths = [Path(npz_path)]

        embs, lbls, spks, crps, fpaths = [], [], [], [], []
        for p in npz_paths:
            data = np.load(p, allow_pickle=True)
            e = data["embeddings"]
            l = data["labels"]
            s = data["speaker_ids"] if "speaker_ids" in data else None
            c = data["corpus"] if "corpus" in data else None
            f = data["file_paths"] if "file_paths" in data else None

            if include_corpora is not None and c is not None:
                mask = np.isin(c, list(include_corpora))
                e = e[mask]
                l = l[mask]
                if s is not None:
                    s = s[mask]
                if c is not None:
                    c = c[mask]
                if f is not None:
                    f = f[mask]

            embs.append(e)
            lbls.append(l)
            if s is not None:
                spks.append(s)
            if c is not None:
                crps.append(c)
            if f is not None:
                fpaths.append(f)

        self.embeddings = np.concatenate(embs, axis=0)
        self.labels = np.concatenate(lbls, axis=0)
        self.speaker_ids = np.concatenate(spks, axis=0) if spks else None
        self.corpus = np.concatenate(crps, axis=0) if crps else None
        self.file_paths = np.concatenate(fpaths, axis=0) if fpaths else None

        if max_samples is not None and max_samples > 0:
            self.embeddings = self.embeddings[:max_samples]
            self.labels = self.labels[:max_samples]
            if self.speaker_ids is not None:
                self.speaker_ids = self.speaker_ids[:max_samples]
            if self.corpus is not None:
                self.corpus = self.corpus[:max_samples]
            if self.file_paths is not None:
                self.file_paths = self.file_paths[:max_samples]

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


