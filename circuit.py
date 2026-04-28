"""
Phase 3 — 23-qubit VQC Topology (Project Pitru-Maraka 2.0).

Wire layout:
    q_sun      [0-4]   : Sun (Karaka for father)       — 5 qubits / nakshatra
    q_ninth    [5-9]   : 9th Lord (house of father)    — 5 qubits / nakshatra
    q_eighth   [10-14] : 8th Lord (house of death)     — 5 qubits / nakshatra
    q_saturn   [15-19] : Saturn (time trigger)         — 5 qubits / nakshatra
    q_ancilla  [20-21] : aggregation registers
    q_target   [22]    : measurement wire

Input encoding:
    RX(bit × π, wire)  replaces BasisState — differentiable, works with any
    PennyLane diff method, and accepts 1-D float tensors directly.

Entanglement topology:
    Law 1 — Karaka meets Time     : CRZ  Sun(0-4)   ↔ Saturn(15-19)
    Law 2 — Father & Death houses : CRZ  9th(5-9)   ↔ 8th(10-14)
    Funnel to ancilla 20          : CRY  first qubit of each group → 20
    Funnel to ancilla 21          : CRY  last  qubit of each group → 21
    Ancilla → target              : CRY  20→22, 21→22

Device strategy:
    lightning.gpu  (WSL/Linux with custatevec) — adjoint diff
                   peak VRAM: 2 × 2^23 × 8 bytes (complex128) ≈ 134 MB
    lightning.qubit (CPU fallback)             — adjoint diff
    default.qubit   (pure-Python fallback)     — parameter-shift
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

N_QUBITS   = 23
N_SPATIAL  = 20   # 4 groups × 5 qubits
N_GROUPS   = 4
ANCILLA_0  = 20
ANCILLA_1  = 21
TARGET     = 22

# 60 trainable parameters total
WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "weights_ry":    (N_SPATIAL,),  # RY on each spatial qubit
    "weights_rx":    (N_SPATIAL,),  # RX on each spatial qubit (trainable layer)
    "weights_law1":  (5,),          # Law 1 CRZ: Sun ↔ Saturn
    "weights_law2":  (5,),          # Law 2 CRZ: 9th ↔ 8th
    "weights_anc0":  (N_GROUPS,),   # CRY first-of-group → ancilla 20
    "weights_anc1":  (N_GROUPS,),   # CRY last-of-group  → ancilla 21
    "weights_target": (2,),         # CRY ancilla → target
}


def _make_device() -> tuple[qml.Device, str]:
    """Try lightning.gpu → lightning.qubit → default.qubit."""
    for name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
        try:
            kwargs: dict = {"wires": N_QUBITS}
            if name != "default.qubit":
                kwargs["c_dtype"] = np.complex64
            dev = qml.device(name, **kwargs)
            print(f"[circuit] PennyLane device: {name}")
            return dev, name
        except Exception as exc:
            print(f"[circuit] {name} unavailable: {exc}")
    raise RuntimeError("No usable PennyLane device found.")


def build_qlayer() -> qml.qnn.TorchLayer:
    """
    Construct the 23-qubit QNode and wrap it in a TorchLayer.

    Input  : float32 tensor (20,)  — 4 nakshatra indices each binary-expanded
                                     to 5 bits, flattened (values 0.0 or 1.0).
    Output : float32 scalar        — E[Z] on wire 22 ∈ [-1, 1].
             Caller converts: P(death=1) = (1 − E[Z]) / 2.
    """
    dev, dev_name = _make_device()
    diff_method = "adjoint" if dev_name in ("lightning.gpu", "lightning.qubit") else "parameter-shift"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def _circuit(
        inputs,
        weights_ry, weights_rx,
        weights_law1, weights_law2,
        weights_anc0, weights_anc1,
        weights_target,
    ):
        # ── A. Input encoding via RX gates ──────────────────────────────────
        # RX(0) = I (bit 0 stays |0⟩), RX(π) ≈ X (bit 1 → |1⟩ up to phase).
        # Differentiable alternative to BasisState; accepts float tensors.
        for i in range(N_SPATIAL):
            qml.RX(inputs[i] * np.pi, wires=i)

        # ── B. Trainable single-qubit rotations ─────────────────────────────
        for i in range(N_SPATIAL):
            qml.RY(weights_ry[i], wires=i)
        for i in range(N_SPATIAL):
            qml.RX(weights_rx[i], wires=i)

        # ── Law 1 — The Karaka meets Time (Sun ↔ Saturn) ────────────────────
        for i in range(5):
            qml.CRZ(weights_law1[i], wires=[i, i + 15])

        # ── Law 2 — House of Father & Death (9th ↔ 8th) ────────────────────
        for i in range(5):
            qml.CRZ(weights_law2[i], wires=[i + 5, i + 10])

        # ── C. Target Funnel ─────────────────────────────────────────────────
        # First qubit of each group (Sun[0], 9th[0], 8th[0], Saturn[0]) → 20
        for k, ctrl in enumerate([0, 5, 10, 15]):
            qml.CRY(weights_anc0[k], wires=[ctrl, ANCILLA_0])

        # Last qubit of each group (Sun[4], 9th[4], 8th[4], Saturn[4]) → 21
        for k, ctrl in enumerate([4, 9, 14, 19]):
            qml.CRY(weights_anc1[k], wires=[ctrl, ANCILLA_1])

        # Ancilla → target
        qml.CRY(weights_target[0], wires=[ANCILLA_0, TARGET])
        qml.CRY(weights_target[1], wires=[ANCILLA_1, TARGET])

        return qml.expval(qml.PauliZ(TARGET))

    return qml.qnn.TorchLayer(_circuit, WEIGHT_SHAPES)
