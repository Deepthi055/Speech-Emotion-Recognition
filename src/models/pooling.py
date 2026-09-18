import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            return torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return hidden_states.mean(dim=1)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # hidden_states: [B, T, D]
        attn_weights = self.attention(hidden_states)  # [B, T, 1]
        
        if attention_mask is not None:
            # Mask padded tokens by setting attention weight to -inf
            # attention_mask is [B, T], 1 for valid, 0 for pad
            mask = attention_mask.unsqueeze(-1).bool()
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
            
        attn_weights = torch.softmax(attn_weights, dim=1)  # [B, T, 1]
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1)  # [B, D]
        return pooled_output
