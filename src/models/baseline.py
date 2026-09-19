import torch
import torch.nn as nn
from src.models.classifier import ClassifierHead

class WavLMClassifier(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 6):
        super().__init__()
        self.classifier_head = ClassifierHead(in_dim=in_dim, num_classes=num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(embeddings)
