"""
Phase 3 — 28-qubit VQC Topology (Project Pitru-Maraka 2.0).

Wire layout:
    q_spatial  [0-24]  : 5 role groups × 5 qubits, loaded from BasisState
    q_ancilla  [25-26] : aggregation / uncomputation registers
    q_target   [27]    : measurement wire

Entanglement topology (sparse, avoids barren plateaus):
    Law 1 — Karaka meets Time     : CRZ  Sun(0-4)      ↔ Saturn(20-24)
    Law 2 — Father & Death houses : CRZ  9th(5-9)      ↔ 8th(10-14)
    Law 3a — Maraka activates 9th : CRZ  Maraka(15-19) → 9th(5-9)
    Law 3b — Maraka activates Sun : CRZ  Maraka(15-19) → Sun(0-4)
    Funnel to ancilla 25          : CRY  first qubit of each role → 25
    Funnel to ancilla 26          : CRY  last  qubit of each role → 26
    Ancilla → target              : CRY  25→27, 26→27

Device strategy:
    lightning.gpu  (WSL/Linux with custatevec) — adjoint diff, 2 state vectors
                   peak VRAM: 2 × 2^28 × 8 bytes (complex64) ≈ 4.2 GB
    lightning.qubit (CPU fallback)             — adjoint diff
    default.qubit   (pure-Python fallback)     — parameter-shift
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

N_QUBITS   = 28
N_SPATIAL  = 25
ANCILLA_0  = 25
ANCILLA_1  = 26
TARGET     = 27

# 82 trainable parameters total
WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "weights_ry":       (N_SPATIAL,),   # RY on each spatial qubit
    "weights_rx":       (N_SPATIAL,),   # RX on each spatial qubit
    "weights_law1":     (5,),           # Law 1 CRZ: Sun ↔ Saturn
    "weights_law2":     (5,),           # Law 2 CRZ: 9th ↔ 8th
    "weights_law3_9":   (5,),           # Law 3a CRZ: Maraka → 9th
    "weights_law3_sun": (5,),           # Law 3b CRZ: Maraka → Sun
    "weights_anc0":     (5,),           # CRY first-of-group → ancilla 25
    "weights_anc1":     (5,),           # CRY last-of-group  → ancilla 26
    "weights_target":   (2,),           # CRY ancilla → target
}


def _make_device() -> tuple[qml.Device, str]:
    """
    Try lightning.gpu → lightning.qubit → default.qubit.
    Uses complex64 to halve state-vector memory
    (2^28 × 8 bytes ≈ 2.1 GB instead of 4.3 GB).
    """
    for name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
        try:
            kwargs = {"wires": N_QUBITS, "c_dtype": np.complex64}
            if name == "default.qubit":
                kwargs.pop("c_dtype")   # default.qubit uses dtype, not c_dtype
            dev = qml.device(name, **kwargs)
            print(f"[circuit] PennyLane device: {name}")
            return dev, name
        except Exception as exc:
            print(f"[circuit] {name} unavailable: {exc}")
    raise RuntimeError("No usable PennyLane device found.")


def build_qlayer() -> qml.qnn.TorchLayer:
    """
    Construct the 28-qubit QNode and wrap it in a TorchLayer.

    Input  : float32 tensor (25,)  — binary transit encoding (one sample).
    Output : float32 tensor (2,)   — [P(|0⟩), P(|1⟩=death)] on wire 27.

    Note: callers must pass one sample at a time (shape (25,)) because
    qml.BasisState requires 1-D input. QMLModel.forward handles the batch loop.
    """
    dev, dev_name = _make_device()

    if dev_name in ("lightning.gpu", "lightning.qubit"):
        diff_method = "adjoint"
    else:
        diff_method = "parameter-shift"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def _circuit(
        inputs,
        weights_ry, weights_rx,
        weights_law1, weights_law2,
        weights_law3_9, weights_law3_sun,
        weights_anc0, weights_anc1,
        weights_target,
    ):
        # ── A. State Preparation ────────────────────────────────────────────
        qml.BasisState(inputs, wires=range(N_SPATIAL))

        # ── B. Single-qubit parameterised rotations ─────────────────────────
        for i in range(N_SPATIAL):
            qml.RY(weights_ry[i], wires=i)
        for i in range(N_SPATIAL):
            qml.RX(weights_rx[i], wires=i)

        # ── Law 1 — The Karaka meets Time ───────────────────────────────────
        for i in range(5):
            qml.CRZ(weights_law1[i], wires=[i, i + 20])

        # ── Law 2 — House of Father & Death ─────────────────────────────────
        for i in range(5):
            qml.CRZ(weights_law2[i], wires=[i + 5, i + 10])

        # ── Law 3a — Maraka activates 9th Lord ──────────────────────────────
        for i in range(5):
            qml.CRZ(weights_law3_9[i], wires=[i + 15, i + 5])

        # ── Law 3b — Maraka activates Sun (Karaka) ──────────────────────────
        for i in range(5):
            qml.CRZ(weights_law3_sun[i], wires=[i + 15, i])

        # ── C. Target Funnel: first qubit of each role → ancilla 25 ─────────
        for k, ctrl in enumerate([0, 5, 10, 15, 20]):
            qml.CRY(weights_anc0[k], wires=[ctrl, ANCILLA_0])

        # Last qubit of each role → ancilla 26
        for k, ctrl in enumerate([4, 9, 14, 19, 24]):
            qml.CRY(weights_anc1[k], wires=[ctrl, ANCILLA_1])

        # Ancilla → target
        qml.CRY(weights_target[0], wires=[ANCILLA_0, TARGET])
        qml.CRY(weights_target[1], wires=[ANCILLA_1, TARGET])

        # qml.probs is not supported by lightning adjoint diff in PennyLane 0.44.
        # qml.expval(PauliZ) is adjoint-compatible; caller converts via:
        #   P(death=1) = (1 − E[Z]) / 2
        return qml.expval(qml.PauliZ(TARGET))

    return qml.qnn.TorchLayer(_circuit, WEIGHT_SHAPES)
