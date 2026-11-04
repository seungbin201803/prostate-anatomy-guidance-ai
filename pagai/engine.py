"""Train / evaluation loops: mean-pool the backbone's tokens, then one classification head."""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .utils import ID_COL
from .tasks import task_loss, predicted_class, compute_metrics, prediction_columns


def autocast(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16" and device.type == "cuda")


def forward_logits(backbone, head, image):
    feat = backbone(image).mean(dim=1).float()  # [B, embed_dim]
    return head(feat).float()


def train_one_epoch(backbone, heads, task, loader, optimizer, scheduler, device, args,
                    writer=None, global_step=0):
    """
    Gradient accumulation over args.grad_accum_steps micro-batches; the scheduler steps once per
    optimizer step. Returns (metrics incl. "loss", global_step).
    """
    backbone.train()
    heads.train()
    head = heads[task]
    accum = max(1, args.grad_accum_steps)
    n_steps = len(loader)
    total_loss, n_batches = 0.0, 0
    all_labels, all_preds, all_logits = [], [], []

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(pbar):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with autocast(device, args.precision):
            logits = forward_logits(backbone, head, image)
            loss = task_loss(logits, label, head.num_classes, head.loss_name)
        (loss / accum).backward()

        if (step + 1) % accum == 0 or step + 1 == n_steps:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(heads.parameters()) + list(backbone.parameters()), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        loss_value = loss.detach().item()
        total_loss += loss_value
        n_batches += 1
        all_labels.append(label.cpu().numpy())
        all_preds.append(predicted_class(logits.detach(), head.loss_name).cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())
        if writer is not None:
            writer.add_scalar("train_step/loss", loss_value, global_step)
            for group in optimizer.param_groups:
                writer.add_scalar(f"train_step/lr_{group['name']}", group["lr"], global_step)
        global_step += 1
        pbar.set_postfix(loss=total_loss / n_batches)

    metrics = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_logits), head.loss_name)
    metrics["loss"] = total_loss / max(1, n_batches)
    return metrics, global_step


@torch.no_grad()
def evaluate(backbone, heads, task, classes, loader, device, precision):
    """
    Predicts every case. Loss and metrics use labeled cases only (label >= 0).
    Returns (metrics incl. "loss", predictions DataFrame).
    """
    backbone.eval()
    heads.eval()
    head = heads[task]
    rows, labels, preds, logits_all = [], [], [], []
    total_loss, n_loss = 0.0, 0

    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with autocast(device, precision):
            logits = forward_logits(backbone, head, image)
        pred = predicted_class(logits, head.loss_name)

        labeled = label >= 0
        if labeled.any():
            total_loss += task_loss(logits[labeled], label[labeled], head.num_classes, head.loss_name).item() * int(labeled.sum())
            n_loss += int(labeled.sum())

        logits_np, pred_np, label_np = logits.cpu().numpy(), pred.cpu().numpy(), label.cpu().numpy()
        for i in range(len(label_np)):
            rows.append({
                ID_COL: batch[ID_COL][i],
                f"{task}_gt": batch["label_raw"][i] if label_np[i] >= 0 else "",
                f"{task}_pred": classes[int(pred_np[i])],
                **prediction_columns(task, classes, logits_np[i], head.loss_name),
            })
            if label_np[i] >= 0:
                labels.append(int(label_np[i]))
                preds.append(int(pred_np[i]))
                logits_all.append(logits_np[i])

    metrics = compute_metrics(labels, preds, np.stack(logits_all) if logits_all else np.zeros((0, 1)), head.loss_name)
    metrics["loss"] = total_loss / n_loss if n_loss else float("nan")
    return metrics, pd.DataFrame(rows)
