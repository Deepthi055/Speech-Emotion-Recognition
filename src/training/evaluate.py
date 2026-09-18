import torch
from src.utils.metrics import compute_metrics


@torch.no_grad()
def evaluate(model, dataloader, device, dry_run: bool = False):
    model.eval()
    all_preds = []
    all_labels = []

    for step, batch in enumerate(dataloader):
        if dry_run and step >= 1:  # Only 1 batch in dry-run
            break
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_values, attention_mask=attention_mask)
        logits = outputs["logits"]

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return compute_metrics(all_labels, all_preds)
