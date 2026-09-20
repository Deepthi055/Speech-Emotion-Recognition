import torch
import torch.nn as nn
from src.models.projection import ProjectionHead
from src.models.classifier import ClassifierHead


class WavLMEmbeddingSupConModel(nn.Module):
    """Model that takes frozen, mean-pooled 768-dim WavLM embeddings

    and projects them via ProjectionHead (for SupCon loss)
    and ClassifierHead (for Cross-Entropy classification).
    """

    def __init__(
        self,
        in_dim: int = 768,
        proj_dim: int = 128,
        num_classes: int = 6,
        use_projection: bool = True,
        use_classifier: bool = True,
    ):
        super().__init__()
        self.use_projection = use_projection
        self.use_classifier = use_classifier

        self.projection_head = (
            ProjectionHead(in_dim=in_dim, out_dim=proj_dim) if use_projection else None
        )
        self.classifier_head = (
            ClassifierHead(in_dim=in_dim, num_classes=num_classes)
            if use_classifier
            else None
        )

    def forward(self, embeddings: torch.Tensor):
        projections = (
            self.projection_head(embeddings)
            if self.projection_head is not None
            else None
        )
        logits = (
            self.classifier_head(embeddings)
            if self.classifier_head is not None
            else None
        )
        return {
            "embeddings": embeddings,
            "projections": projections,
            "logits": logits,
        }
