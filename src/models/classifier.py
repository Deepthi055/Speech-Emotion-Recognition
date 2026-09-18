import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 6):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
