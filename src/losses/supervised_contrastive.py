import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        weights_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        if features.dim() == 2:
            features = features.unsqueeze(1)

        num_views = features.shape[1]
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_features = contrast_features
        anchor_count = num_views

        similarity = torch.div(
            torch.matmul(anchor_features, contrast_features.T),
            self.temperature,
        )

        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(similarity),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )

        if weights_mask is not None:
            mask = weights_mask.float().to(device)
            if mask.shape != similarity.shape:
                mask = mask.repeat(anchor_count, anchor_count)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            mask = mask.repeat(anchor_count, anchor_count)
        else:
            raise ValueError("Either labels or weights_mask must be provided.")

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        valid_anchors = (mask.sum(1) > 0).float()
        loss = (loss * valid_anchors).sum() / (valid_anchors.sum() + 1e-12)

        return loss
