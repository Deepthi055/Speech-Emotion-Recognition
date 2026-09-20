import torch


def compute_standard_weights(labels: torch.Tensor) -> torch.Tensor:
    """Standard Supervised Contrastive weights (same emotion = positive)."""
    same_emotion = labels.unsqueeze(1) == labels.unsqueeze(0)
    weights = same_emotion.float()
    weights.fill_diagonal_(0.0)
    return weights


def compute_speaker_aware_weights(
    labels: torch.Tensor, speaker_ids: torch.Tensor
) -> torch.Tensor:
    """Step 8: Speaker-Aware SupCon weights.
    Same emotion + different speaker = 1.0 (preferred positive)
    Same emotion + same speaker = 0.25 (weaker positive to discourage speaker shortcut)
    """
    same_emotion = labels.unsqueeze(1) == labels.unsqueeze(0)
    same_speaker = speaker_ids.unsqueeze(1) == speaker_ids.unsqueeze(0)

    weights = torch.zeros(
        labels.size(0), labels.size(0), device=labels.device, dtype=torch.float32
    )
    weights[same_emotion & ~same_speaker] = 1.0
    weights[same_emotion & same_speaker] = 0.25
    weights.fill_diagonal_(0.0)
    return weights


def compute_corpus_aware_weights(
    labels: torch.Tensor, corpus_ids: torch.Tensor
) -> torch.Tensor:
    """Step 9: Corpus-Aware SupCon weights.
    Same emotion + different corpus = 1.0 (preferred positive to remove corpus bias)
    Same emotion + same corpus = 0.25 (weaker positive)
    """
    same_emotion = labels.unsqueeze(1) == labels.unsqueeze(0)
    same_corpus = corpus_ids.unsqueeze(1) == corpus_ids.unsqueeze(0)

    weights = torch.zeros(
        labels.size(0), labels.size(0), device=labels.device, dtype=torch.float32
    )
    weights[same_emotion & ~same_corpus] = 1.0
    weights[same_emotion & same_corpus] = 0.25
    weights.fill_diagonal_(0.0)
    return weights


def compute_speaker_corpus_aware_weights(
    labels: torch.Tensor, speaker_ids: torch.Tensor, corpus_ids: torch.Tensor
) -> torch.Tensor:
    """Step 10: Proposed Speaker + Corpus-Aware SupCon weights.
    Same emotion + cross-corpus & diff speaker = 1.0 (highest priority positive)
    Same emotion + intra-corpus & diff speaker = 0.5 (medium positive)
    Same emotion + same speaker = 0.25 (lowest positive)
    """
    same_emotion = labels.unsqueeze(1) == labels.unsqueeze(0)
    same_speaker = speaker_ids.unsqueeze(1) == speaker_ids.unsqueeze(0)
    same_corpus = corpus_ids.unsqueeze(1) == corpus_ids.unsqueeze(0)

    weights = torch.zeros(
        labels.size(0), labels.size(0), device=labels.device, dtype=torch.float32
    )
    weights[same_emotion & ~same_corpus & ~same_speaker] = 1.0
    weights[same_emotion & same_corpus & ~same_speaker] = 0.5
    weights[same_emotion & same_speaker] = 0.25
    weights.fill_diagonal_(0.0)
    return weights
