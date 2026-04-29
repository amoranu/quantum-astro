"""
Phase 4 — QML Training Loop (Project Pitru-Maraka 2.0).

Wraps the PennyLane TorchLayer in a thin nn.Module, trains it with
BCEWithLogitsLoss + Adam + CosineAnnealingLR, and saves a checkpoint.

The model output is logit = α · E[Z] + β, where (α, β) are trainable
classical parameters. Probability is recovered as σ(logit). Without this
readout-bias, the circuit alone cannot push P past ~0.5 confidently for
either class because E[Z] tends to stay near zero.

Eval is reported on a SEPARATE held-out test set (subject-level split in
pipeline.py) to detect memorization. If train_acc rises while test_acc
stagnates near 0.5, the model is memorizing rather than generalizing.

23-qubit state vectors are small (~8 MB each), so VRAM is not a constraint.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from circuit import build_qlayer


class QMLModel(nn.Module):
    """
    16-qubit PennyLane TorchLayer + classical Linear(3, 1) head.

    Forward input  : (batch, 13) float — angle features (radians).
    Quantum output : (batch, 3)  float — [⟨Z⟩, ⟨X⟩, ⟨Y⟩] on the target wire.
    Forward output : (batch,)   float — *logits* (apply σ for probability).

    Z-init: head.weight = [-1, 0, 0], head.bias = 0 — preserves v6's α=-1 init
    behaviour at start so the first epoch numbers are comparable across versions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.qlayer = build_qlayer()
        self.head   = nn.Linear(3, 1)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[-1.0, 0.0, 0.0]]))
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        expvals = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])])  # (B, 3)
        return self.head(expvals).squeeze(-1)  # (B,) logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_split(model: nn.Module, X: torch.Tensor, Y_flat: torch.Tensor) -> dict:
    model.eval()
    logits = model(X)
    probs  = torch.sigmoid(logits)
    pred   = (probs >= 0.5).float()
    acc    = (pred == Y_flat).float().mean().item()
    pos_mask = (Y_flat == 1)
    neg_mask = ~pos_mask
    acc_pos = pred[pos_mask].mean().item() if pos_mask.any() else 0.0
    acc_neg = (1.0 - pred[neg_mask]).mean().item() if neg_mask.any() else 0.0
    return {"acc": acc, "acc_pos": acc_pos, "acc_neg": acc_neg}


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test:  torch.Tensor | None = None,
    Y_test:  torch.Tensor | None = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.01,
    save_path: str | Path = "model.pt",
) -> QMLModel:
    """
    Train the QML model with held-out test eval and CosineAnnealingLR.

    Args:
        X_train, Y_train : training tensors (N_tr, 13) and (N_tr, 1).
        X_test, Y_test   : held-out test tensors (subject-disjoint from train).
        epochs           : number of full passes over the training set.
        batch_size       : samples per gradient step.
        lr               : initial Adam learning rate; decays via cosine.
        save_path        : checkpoint path.
    """
    y_train_flat = Y_train.squeeze(1)
    y_test_flat  = Y_test.squeeze(1) if Y_test is not None else None
    dataset = TensorDataset(X_train, y_train_flat)

    # Class-balanced sampler over training set
    n_pos = int(y_train_flat.sum().item())
    n_neg = len(y_train_flat) - n_pos
    w_pos = 1.0 / max(n_pos, 1)
    w_neg = 1.0 / max(n_neg, 1)
    sample_weights = torch.where(y_train_flat == 1, w_pos, w_neg)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(y_train_flat), replacement=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)

    model     = QMLModel()
    criterion = nn.BCEWithLogitsLoss()
    # AdamW (weight_decay) — penalises memorisation by shrinking unused weight
    # magnitudes; counteracts the unbounded growth of α we saw in v6 e20.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    n_params = sum(p.numel() for p in model.parameters())
    has_test = X_test is not None and Y_test is not None
    print(
        f"\n[train] train: {len(dataset)} samples ({n_pos} pos / {n_neg} neg)"
        + (f" | test: {len(X_test)} samples ({int(y_test_flat.sum())} pos / {len(y_test_flat) - int(y_test_flat.sum())} neg)" if has_test else "")
        + f"\n[train] {epochs} epochs | batch={batch_size} | lr={lr} (cosine→{lr*0.05:.4f}) | params={n_params}"
        f"\n[train] WeightedRandomSampler active — batches ~50/50 balanced\n",
        flush=True,
    )

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss   = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss   = epoch_loss / max(n_batches, 1)
        cur_lr     = scheduler.get_last_lr()[0]
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            tr = _eval_split(model, X_train, y_train_flat)
            line = (
                f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  lr={cur_lr:.4f}  "
                f"TRAIN acc={tr['acc']:.3f} pos={tr['acc_pos']:.3f} neg={tr['acc_neg']:.3f}"
            )
            if has_test:
                te = _eval_split(model, X_test, y_test_flat)
                line += f"  TEST acc={te['acc']:.3f} pos={te['acc_pos']:.3f} neg={te['acc_neg']:.3f}"
                history.append({"epoch": epoch, "loss": avg_loss, "train": tr, "test": te,
                                "head_w": model.head.weight.detach().clone().tolist(),
                                "head_b": model.head.bias.item()})
            else:
                history.append({"epoch": epoch, "loss": avg_loss, "train": tr,
                                "head_w": model.head.weight.detach().clone().tolist(),
                                "head_b": model.head.bias.item()})
            w = model.head.weight.flatten()
            line += f"  head=[{w[0].item():+.3f},{w[1].item():+.3f},{w[2].item():+.3f}]+{model.head.bias.item():+.3f}"
            print(line, flush=True)
        else:
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  lr={cur_lr:.4f}", flush=True)

    save_path = Path(save_path)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epochs":      epochs,
            "history":     history,
            "n_params":    n_params,
        },
        save_path,
    )
    print(f"\n[train] Checkpoint saved → {save_path}", flush=True)

    model.eval()
    return model


def load_model(model_path: str | Path = "model.pt") -> QMLModel:
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    model      = QMLModel()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[train] Loaded model from {model_path}  (trained {checkpoint['epochs']} epochs)")
    return model
