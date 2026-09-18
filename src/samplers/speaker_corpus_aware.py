import random
from collections import defaultdict
import torch
from torch.utils.data import Sampler


def compute_speaker_corpus_weights(labels: torch.Tensor, speaker_ids: torch.Tensor, corpus_ids: torch.Tensor) -> torch.Tensor:
    labels = labels.view(-1, 1)
    speaker_ids = speaker_ids.view(-1, 1)
    corpus_ids = corpus_ids.view(-1, 1)

    same_emotion = torch.eq(labels, labels.T)
    diff_speaker = torch.ne(speaker_ids, speaker_ids.T)
    diff_corpus = torch.ne(corpus_ids, corpus_ids.T)
    same_speaker_and_corpus = (~diff_speaker) & (~diff_corpus)

    weights = torch.zeros_like(same_emotion, dtype=torch.float32)

    preferred_positives = same_emotion & (diff_speaker & diff_corpus)
    semi_preferred_positives = same_emotion & (diff_speaker | diff_corpus) & (~preferred_positives)
    same_spk_corp_positives = same_emotion & same_speaker_and_corpus

    weights[preferred_positives] = 1.0
    weights[semi_preferred_positives] = 0.8
    weights[same_spk_corp_positives] = 0.2

    return weights


class SpeakerCorpusAwareBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.emotion_spk_corp_indices = defaultdict(list)
        for idx in range(len(dataset)):
            row = dataset.df.iloc[idx]
            key = (row["emotion"], row.get("corpus", "default"), row["speaker_id"])
            self.emotion_spk_corp_indices[key].append(idx)

        self.emotions = sorted(list(set(row["emotion"] for _, row in dataset.df.iterrows())))

    def __iter__(self):
        grouped = {k: list(v) for k, v in self.emotion_spk_corp_indices.items()}
        for k in grouped:
            random.shuffle(grouped[k])

        batches = []
        current_batch = []

        keys = list(grouped.keys())
        random.shuffle(keys)

        while any(len(v) > 0 for v in grouped.values()):
            selected_keys = [k for k in keys if len(grouped[k]) > 0]
            if not selected_keys:
                break

            for key in selected_keys:
                if len(grouped[key]) > 0:
                    current_batch.append(grouped[key].pop())
                    if len(current_batch) == self.batch_size:
                        batches.append(current_batch)
                        current_batch = []

        if len(current_batch) > 0 and not self.drop_last:
            batches.append(current_batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
