"""
Phase 3 — 16-qubit angle-encoded VQC (Project Pitru-Maraka 2.0, v6).

Wire layout (1 qubit per entity, continuous angle encoding):
    q[0]  Sun longitude
    q[1]  Moon longitude
    q[2]  Mars longitude
    q[3]  Mercury longitude
    q[4]  Jupiter longitude
    q[5]  Venus longitude
    q[6]  Saturn longitude
    q[7]  Rahu longitude (mean north node)
    q[8]  Ketu longitude (Rahu + 180°)
    q[9]  Natal lagna sign  (0-11 → 2π)
    q[10] Active Mahadasha lord (0-8 → 2π)
    q[11] Active Antardasha lord (0-8 → 2π)
    q[12] Natal 8th house cusp longitude
    q[13] Ancilla 0 (aggregator)
    q[14] Ancilla 1 (aggregator)
    q[15] Target

Why this design:
  * 1-qubit-per-entity at degree precision packs ~5.8× more information
    than v5's 4×5-bit nakshatra encoding (20 bits total).
  * Adds the 6 missing planets (Moon, Mars, Mercury, Jupiter, Venus,
    Rahu, Ketu) — Rahu/Ketu are *the* primary maraka indicators in
    Jyotish but were entirely absent from v5.
  * Adds dasha state (MD/AD lords). Dashas gate which transits "fire";
    omitting them was a structural missing-input bug, not a tuning issue.
  * State vector is 2^16 × 8 bytes ≈ 0.5 MB — orders of magnitude smaller
    than v5's 67 MB, so per-sample circuit cost drops ~10×.

Trainable structure:
  * 2 layers of (RY, RX) per entity qubit — chart-aware variational depth
  * CRX entanglement chain: q[i] ↔ q[i+1] for i=0..11 (12 gates)
  * Full CRY funnel: every entity → both ancillas (26 gates)
  * CRY ancilla → target (2 gates)
  * Multi-Pauli readout: ⟨Z⟩, ⟨X⟩, ⟨Y⟩ on target — Z reads populations,
    X/Y read coherence (the phase encoded by the v6 RZ symmetry-breaker).
  * Classical Linear(3, 1) head in train.py mixes the 3 expectations.

Total quantum params: 13×4 + 12 + 26 + 2 = 92.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

N_QUBITS    = 16
N_ENTITIES  = 13
ANCILLA_0   = 13
ANCILLA_1   = 14
TARGET      = 15

WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "weights_ry1":   (N_ENTITIES,),       # single variational layer: RY
    "weights_rx1":   (N_ENTITIES,),       # single variational layer: RX
    "weights_chain": (N_ENTITIES - 1,),   # CRX entity_i ↔ entity_{i+1}
    "weights_anc0":  (N_ENTITIES,),       # CRY entity → ancilla 0
    "weights_anc1":  (N_ENTITIES,),       # CRY entity → ancilla 1
    "weights_target": (2,),               # CRY ancilla → target
}


def _make_device() -> tuple[qml.Device, str]:
    """Try lightning.gpu → lightning.qubit → default.qubit."""
    for name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
        try:
            kwargs: dict = {"wires": N_QUBITS}
            if name != "default.qubit":
                kwargs["c_dtype"] = np.complex64
            dev = qml.device(name, **kwargs)
            print(f"[circuit] PennyLane device: {name}", flush=True)
            return dev, name
        except Exception as exc:
            print(f"[circuit] {name} unavailable: {exc}", flush=True)
    raise RuntimeError("No usable PennyLane device found.")


def build_qlayer() -> qml.qnn.TorchLayer:
    """
    16-qubit angle-encoded QNode wrapped as a TorchLayer.

    Input  : float32 tensor (13,)   — radians.
    Output : float32 tensor (3,)    — [⟨Z⟩, ⟨X⟩, ⟨Y⟩] on target wire, each ∈ [-1, 1].
             Caller mixes via Linear(3, 1) → logit → σ for P(death).
    """
    dev, dev_name = _make_device()
    diff_method = "adjoint" if dev_name in ("lightning.gpu", "lightning.qubit") else "parameter-shift"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def _circuit(
        inputs,
        weights_ry1, weights_rx1,
        weights_chain,
        weights_anc0, weights_anc1,
        weights_target,
    ):
        # ── A. Continuous angle encoding (1 qubit per entity) ─────────────
        # RX produces |populations| symmetric in θ ↔ 2π-θ. Adding RZ(θ)
        # encodes phase that breaks the symmetry — downstream CRX / CRY
        # gates extract phase as population, so longitudes 100° and 260°
        # (or lagnas 1 and 11) become genuinely distinguishable.
        for i in range(N_ENTITIES):
            qml.RX(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)

        # ── B. Single variational layer (v9: dropped 2nd layer to reduce
        # capacity — v8 had 92 quantum + 4 classical params over 220 unique
        # positives, ~2 samples per param. Halving the variational depth
        # gives ~4 samples per param.) ────────────────────────────────────
        for i in range(N_ENTITIES):
            qml.RY(weights_ry1[i], wires=i)
        for i in range(N_ENTITIES):
            qml.RX(weights_rx1[i], wires=i)

        # ── C. CRX entanglement chain — propagates info across all entities
        for i in range(N_ENTITIES - 1):
            qml.CRX(weights_chain[i], wires=[i, i + 1])

        # ── E. Full funnel: every entity → both ancillas ──────────────────
        for i in range(N_ENTITIES):
            qml.CRY(weights_anc0[i], wires=[i, ANCILLA_0])
        for i in range(N_ENTITIES):
            qml.CRY(weights_anc1[i], wires=[i, ANCILLA_1])

        # Ancilla → target
        qml.CRY(weights_target[0], wires=[ANCILLA_0, TARGET])
        qml.CRY(weights_target[1], wires=[ANCILLA_1, TARGET])

        # Multi-Pauli readout on the target qubit. Z reads populations;
        # X/Y read coherence introduced by the RZ phase encoding in step A.
        return [
            qml.expval(qml.PauliZ(TARGET)),
            qml.expval(qml.PauliX(TARGET)),
            qml.expval(qml.PauliY(TARGET)),
        ]

    return qml.qnn.TorchLayer(_circuit, WEIGHT_SHAPES)
