import torch
import torch.nn as nn
from .wavlm import WavLMBackbone
from .projection import ProjectionHead
from .classifier import ClassifierHead


class WavLMSupConModel(nn.Module):
    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base",
        num_classes: int = 6,
        proj_dim: int = 128,
        use_projection: bool = True,
        use_classifier: bool = True,
        pooling: str = "attention",
    ):
        super().__init__()
        self.backbone = WavLMBackbone(model_id=model_id, pooling=pooling)
        self.projection_head = ProjectionHead(in_dim=768, out_dim=proj_dim) if use_projection else None
        self.classifier_head = ClassifierHead(in_dim=768, num_classes=num_classes) if use_classifier else None

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        embeddings = self.backbone(input_values, attention_mask=attention_mask)
        projections = self.projection_head(embeddings) if self.projection_head is not None else None
        logits = self.classifier_head(embeddings) if self.classifier_head is not None else None
        return {
            "embeddings": embeddings,
            "projections": projections,
            "logits": logits,
        }
