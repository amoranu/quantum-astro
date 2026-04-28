"""
Phase 4 — QML Training Loop (Project Pitru-Maraka 2.0).

Wraps the PennyLane TorchLayer in a thin nn.Module, trains it with
BCEWithLogitsLoss + Adam, and saves a checkpoint to disk.

The model output is logit = α · E[Z] + β, where (α, β) are trainable
classical parameters. Probability is recovered as σ(logit). Without this
readout-bias, the circuit alone cannot push P past ~0.5 confidently for
either class because E[Z] tends to stay near zero.

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
    23-qubit PennyLane TorchLayer + classical readout (α · E[Z] + β).

    Forward input  : (batch, 20) float — binary transit encoding.
    Forward output : (batch,)   float — *logits* (apply σ for probability).

    α is initialised to -1 so that σ(α·E[Z] + β) at β=0, E[Z]=0 starts at 0.5,
    and a negative E[Z] (typical "death" signal direction) maps to higher P.
    """

    def __init__(self) -> None:
        super().__init__()
        self.qlayer = build_qlayer()
        self.alpha = nn.Parameter(torch.tensor(-1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TorchLayer 0.44 passes full batch; loop so each call receives (20,).
        if x.dim() == 1:
            x = x.unsqueeze(0)
        expvals = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])])  # (B,)
        return self.alpha * expvals + self.beta  # logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train(
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int = 150,
    batch_size: int = 16,
    lr: float = 0.01,
    save_path: str | Path = "model.pt",
) -> QMLModel:
    """
    Train the QML model and save a checkpoint.

    Args:
        X          : (N, 20) float tensor — binary transit encodings.
        Y          : (N, 1)  float tensor — labels (1=death, 0=baseline).
        epochs     : number of full passes over the dataset.
        batch_size : samples per gradient step (16 = VRAM-safe default).
        lr         : Adam learning rate.
        save_path  : where to write the checkpoint.

    Returns:
        The trained QMLModel (weights frozen after return).
    """
    y_flat   = Y.squeeze(1)                             # (N,)
    dataset  = TensorDataset(X, y_flat)

    # WeightedRandomSampler — balance positive vs negative classes per batch.
    # Without this the model collapses to "always predict 0" (91.9% accuracy
    # on the 8.1%-positive dataset).
    n_pos = int(y_flat.sum().item())
    n_neg = len(y_flat) - n_pos
    w_pos = 1.0 / max(n_pos, 1)
    w_neg = 1.0 / max(n_neg, 1)
    sample_weights = torch.where(y_flat == 1, w_pos, w_neg)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(y_flat), replacement=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)

    model     = QMLModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n[train] {len(dataset)} samples ({n_pos} pos / {n_neg} neg) | "
        f"{epochs} epochs | batch={batch_size} | lr={lr} | params={n_params}\n"
        f"[train] WeightedRandomSampler active — batches ~50/50 balanced\n",
        flush=True,
    )

    history: list[float] = []

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

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                all_logits = model(X)
                all_pred   = torch.sigmoid(all_logits)
                acc = ((all_pred >= 0.5).float() == y_flat).float().mean().item()
                pos_mask = (y_flat == 1)
                neg_mask = ~pos_mask
                acc_pos = ((all_pred[pos_mask] >= 0.5).float()).mean().item() if pos_mask.any() else 0.0
                acc_neg = ((all_pred[neg_mask] <  0.5).float()).mean().item() if neg_mask.any() else 0.0
                a, b = model.alpha.item(), model.beta.item()
            print(
                f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                f"acc={acc:.4f}  pos={acc_pos:.3f}  neg={acc_neg:.3f}  "
                f"α={a:+.3f} β={b:+.3f}",
                flush=True,
            )
        else:
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}", flush=True)

    # ── Save checkpoint ────────────────────────────────────────────────────
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
    """Load a trained QMLModel from a checkpoint."""
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    model      = QMLModel()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[train] Loaded model from {model_path}  (trained {checkpoint['epochs']} epochs)")
    return model
