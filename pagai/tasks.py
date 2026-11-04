"""
Classification head, loss, prediction and metrics per --loss. For a K-class task (class index =
position in the class list):

    coral  CORAL ordinal head (Cao et al. 2020, coral-pytorch): one shared weight vector plus K-1
           biases -> K-1 cumulative logits, logit_k = logit P(y > k). Loss: coral_loss over the
           extended binary labels. Prediction: number of k with P(y > k) > 0.5. Used for PI-RADS.
    ce     Linear -> K logits, softmax cross-entropy, prediction = argmax. Used for binary csPCa.

Metrics: accuracy, mae/mse (in class steps), quadratic weighted kappa (qwk), and auc for a binary
task. The checkpoint-selection score is auc for a binary task and qwk otherwise.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label

LOSS_CHOICES = ("coral", "ce")


class TaskHead(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=None, dropout=0.1, loss_name="coral"):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"a classification head needs at least 2 classes, got {num_classes}")
        hidden_dim = hidden_dim or embed_dim // 2
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc = CoralLayer(size_in=hidden_dim, num_classes=num_classes) if loss_name == "coral" \
            else nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc(self.net(x))  # [B, K-1] (coral) / [B, K] (ce)


def build_heads(embed_dim, task, num_classes, hidden_dim, dropout, loss_name):
    """A ModuleDict keyed by the task name -- the checkpoint layout of the original research code."""
    return nn.ModuleDict({task: TaskHead(embed_dim, num_classes, hidden_dim, dropout, loss_name)})


def task_loss(logits, target, num_classes, loss_name):
    if loss_name == "ce":
        return F.cross_entropy(logits, target)
    levels = levels_from_labelbatch(target.cpu(), num_classes=num_classes).to(logits.device, logits.dtype)
    return coral_loss(logits, levels)


def predicted_class(logits, loss_name):
    return logits.argmax(1) if loss_name == "ce" else proba_to_label(torch.sigmoid(logits))


def logits_to_probs(logits, loss_name):
    """ce: softmax, P(y = k) [.., K]; coral: sigmoid, P(y > k) [.., K-1]."""
    return torch.softmax(logits, -1) if loss_name == "ce" else torch.sigmoid(logits)


def compute_metrics(labels, preds, logits, loss_name):
    labels, preds = np.asarray(labels), np.asarray(preds)
    if labels.size == 0:
        return {"n": 0}
    metrics = {
        "n": int(labels.size),
        "accuracy": float((preds == labels).mean()),
        "mae": float(np.abs(preds - labels).mean()),
        "mse": float(((preds - labels) ** 2).mean()),
    }
    try:
        metrics["qwk"] = float(cohen_kappa_score(labels, preds, weights="quadratic"))
    except ValueError:
        metrics["qwk"] = float("nan")
    if np.isnan(metrics["qwk"]):  # sklearn returns nan (with a warning) when undefined
        metrics["qwk"] = float("nan")

    logits = np.asarray(logits)
    if logits.shape[1] == (2 if loss_name == "ce" else 1):  # binary task
        score = logits[:, 1] - logits[:, 0] if loss_name == "ce" else logits[:, 0]  # monotonic in P(y = 1)
        try:
            metrics["auc"] = float(roc_auc_score(labels, score))
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics


def selection_score(metrics, num_classes):
    """Higher is better: auc for a binary task, qwk for a multi-class one (nan if undefined)."""
    return metrics.get("auc" if num_classes == 2 else "qwk", float("nan"))


def format_metrics(metrics):
    if not metrics.get("n"):
        return "n=0"
    keys = ["accuracy", "mae", "mse", "qwk"] + (["auc"] if "auc" in metrics else [])
    return f"n={metrics['n']} " + " ".join(f"{k}={metrics[k]:.4f}" for k in keys)


def prediction_columns(task, classes, logits, loss_name):
    """
    Per-sample probability columns for the predictions CSV:
        ce     {task}_prob_<c>     P(y = c), every class
        coral  {task}_prob_gt_<c>  P(y > c), every class but the last
    """
    probs = logits_to_probs(torch.as_tensor(np.asarray(logits, dtype=np.float32)), loss_name).numpy()
    if loss_name == "ce":
        return {f"{task}_prob_{c}": float(probs[k]) for k, c in enumerate(classes)}
    return {f"{task}_prob_gt_{c}": float(probs[k]) for k, c in enumerate(classes[:-1])}
