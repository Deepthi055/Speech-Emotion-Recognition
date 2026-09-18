import torch
import torch.nn as nn
from transformers import WavLMModel

from src.models.pooling import MeanPooling, AttentionPooling


class WavLMBackbone(nn.Module):
    def __init__(self, model_id: str = "microsoft/wavlm-base", freeze_feature_extractor: bool = True, pooling: str = "attention"):
        super().__init__()
        self.model = WavLMModel.from_pretrained(model_id)
        if freeze_feature_extractor:
            self.model.feature_extractor.eval()
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
                
        # The hidden size for wavlm-base is 768. 
        # If a different model is used (e.g. large), it might be 1024, but we assume 768 for base models.
        hidden_size = self.model.config.hidden_size
        
        if pooling == "attention":
            self.pooler = AttentionPooling(in_dim=hidden_size)
        elif pooling == "mean":
            self.pooler = MeanPooling()
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        
        # The feature extractor downsamples the audio. We need to downsample the attention mask as well.
        if attention_mask is not None:
            attention_mask = self.model._get_feature_vector_attention_mask(
                outputs.last_hidden_state.shape[1], attention_mask, add_adapter=False
            )
            
        return self.pooler(outputs.last_hidden_state, attention_mask=attention_mask)
