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
    Law 1 — Karaka meets Time     : CRX  Sun(0-4)   ↔ Saturn(15-19)
    Law 2 — Father & Death houses : CRX  9th(5-9)   ↔ 8th(10-14)
    Full funnel → ancilla 20      : CRY  every spatial qubit (0-19) → 20
    Full funnel → ancilla 21      : CRY  every spatial qubit (0-19) → 21
    Ancilla → target              : CRY  20→22, 21→22

Why CRX (not CRZ): CRZ is diagonal in the Z-basis and commutes with the
final qml.expval(qml.PauliZ) measurement, so phase-only entanglement does
not propagate to the readout. CRX changes populations and does propagate.

Why full funnel: the previous {first, last}-of-group funnel only routed
8 of 20 spatial qubits to the ancilla. Middle qubits (1,2,3, 6,7,8, ...)
contributed nothing to the gradient, leaving 12 of 20 input bits
effectively masked out.

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

# 92 trainable quantum parameters total (+2 classical α, β in train.py)
WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "weights_ry":    (N_SPATIAL,),  # RY on each spatial qubit
    "weights_rx":    (N_SPATIAL,),  # RX on each spatial qubit
    "weights_law1":  (5,),          # Law 1 CRX: Sun ↔ Saturn
    "weights_law2":  (5,),          # Law 2 CRX: 9th ↔ 8th
    "weights_anc0":  (N_SPATIAL,),  # CRY every spatial qubit → ancilla 20
    "weights_anc1":  (N_SPATIAL,),  # CRY every spatial qubit → ancilla 21
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
        # CRX (not CRZ): population-changing entanglement that propagates to
        # the final Z measurement.
        for i in range(5):
            qml.CRX(weights_law1[i], wires=[i, i + 15])

        # ── Law 2 — House of Father & Death (9th ↔ 8th) ────────────────────
        for i in range(5):
            qml.CRX(weights_law2[i], wires=[i + 5, i + 10])

        # ── C. Full target funnel ───────────────────────────────────────────
        # Every spatial qubit feeds both ancillas — so every input bit has
        # a path to the readout. (Previous {first,last}-only funnel masked
        # 12 of 20 input bits.)
        for i in range(N_SPATIAL):
            qml.CRY(weights_anc0[i], wires=[i, ANCILLA_0])
        for i in range(N_SPATIAL):
            qml.CRY(weights_anc1[i], wires=[i, ANCILLA_1])

        # Ancilla → target
        qml.CRY(weights_target[0], wires=[ANCILLA_0, TARGET])
        qml.CRY(weights_target[1], wires=[ANCILLA_1, TARGET])

        return qml.expval(qml.PauliZ(TARGET))

    return qml.qnn.TorchLayer(_circuit, WEIGHT_SHAPES)
