"""
Phase 4 — QML Training Loop (Project Pitru-Maraka 2.0).

Wraps the PennyLane TorchLayer in a thin nn.Module, trains it with
BCELoss + Adam, and saves a checkpoint to disk.

Batch size is fixed at 16 to stay within the RTX 4060 8 GB VRAM budget
(each 28-qubit adjoint-diff run consumes ~4.2 GB peak; sequential per-sample
processing inside TorchLayer means only one state vector is live at a time).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from circuit import build_qlayer


class QMLModel(nn.Module):
    """
    Thin wrapper around the 28-qubit PennyLane TorchLayer.

    Forward input  : (batch, 25) float — binary transit encoding.
    Forward output : (batch,)   float — P(father death) in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.qlayer = build_qlayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BasisState requires 1-D input; TorchLayer 0.44 passes the full batch.
        # Loop so each qlayer call receives (25,) → returns scalar E[Z] ∈ [-1, 1].
        # Convert expectation value to death probability: P(1) = (1 − E[Z]) / 2
        if x.dim() == 1:
            x = x.unsqueeze(0)
        expvals = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])])  # (B,)
        return (1.0 - expvals) / 2.0  # (B,) — P(death=1) ∈ [0, 1]


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
        X          : (N, 25) float tensor — binary transit encodings.
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
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model     = QMLModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n[train] {len(dataset)} samples | {epochs} epochs | "
        f"batch={batch_size} | lr={lr} | params={n_params}\n"
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

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                all_pred = model(X)
                acc = ((all_pred >= 0.5).float() == y_flat).float().mean().item()
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

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
    print(f"\n[train] Checkpoint saved → {save_path}")

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
