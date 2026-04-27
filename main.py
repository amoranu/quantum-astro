"""
Project Pitru-Maraka 2.0 — End-to-end runner.

Usage:
    # Full run (build dataset + train + demo inference):
    python main.py

    # Skip training if a checkpoint already exists:
    python main.py --skip-train

    # Custom data / model paths:
    python main.py --data /path/to/candidates.json --model my_model.pt

    # Training hyperparameters:
    python main.py --epochs 200 --batch-size 16 --lr 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

DEFAULT_DATA = (
    _HERE.parent
    / "neuro-symbolic-astro"
    / "astroql"
    / "applications"
    / "father_longevity"
    / "data"
    / "candidates_post1970.json"
)

DEFAULT_MODEL = _HERE / "model.pt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Project Pitru-Maraka 2.0")
    parser.add_argument("--data",       default=str(DEFAULT_DATA), help="Path to candidates JSON")
    parser.add_argument("--model",      default=str(DEFAULT_MODEL), help="Path to save / load model")
    parser.add_argument("--skip-train", action="store_true",        help="Skip training, load existing model")
    parser.add_argument("--epochs",     type=int,   default=150,    help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=16,     help="Batch size (VRAM constraint)")
    parser.add_argument("--lr",         type=float, default=0.01,   help="Adam learning rate")
    parser.add_argument("--target-size",type=int,   default=1000,   help="Approximate dataset size")
    args = parser.parse_args()

    data_path  = Path(args.data)
    model_path = Path(args.model)

    print("=" * 60)
    print("  Project Pitru-Maraka 2.0 — Dynamic Quantum Astrology VQE")
    print("=" * 60)

    # ── Phase 2: Build dataset ─────────────────────────────────────────────
    if not args.skip_train:
        print(f"\n[Phase 2] Building dataset from:\n  {data_path}\n")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        from pipeline import build_dataset
        X, Y = build_dataset(data_path, target_size=args.target_size)
        print(f"\n  X: {tuple(X.shape)}  Y: {tuple(Y.shape)}")
        print(f"  Class balance: {Y.mean().item():.3f} (fraction positive)\n")

        # ── Phases 3+4: Train ──────────────────────────────────────────────
        print("[Phases 3+4] Initialising 28-qubit VQC and training …\n")
        from train import train
        model = train(
            X, Y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=model_path,
        )
    else:
        print(f"\n[skip] Loading existing model from {model_path}")
        from train import load_model
        model = load_model(model_path)

    # ── Phase 5: Demo inference ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 5 — Demo Inference")
    print("=" * 60)

    from predict import predict, scan_window

    # Example 1: Dale Earnhardt Jr. (father died 2001-02-18)
    demo_birth = {
        "birth_date": "1974-10-10",
        "birth_time": "05:33",
        "lat":  35.4167,
        "lon":  -80.5833,
        "tz":   "America/New_York",
    }

    print("\n─── Single-date query ───────────────────────────────")
    result = predict(demo_birth, "2001-02-18", model)
    print(f"  Subject   : Dale Earnhardt Jr.")
    print(f"  Date      : {result['target_date']}")
    print(f"  Roles     : {result['roles']}")
    print(f"  Nakshatras: {result['nakshatras']}")
    print(f"  Result    : {result['prediction']}  (P={result['probability']:.4f})")

    # Example 2: Scan a 1-year window around the death date
    print("\n─── Weekly scan  2000-08-01 → 2001-08-01  ──────────")
    hotspots = scan_window(demo_birth, "2000-08-01", "2001-08-01", model, step_days=7)
    if hotspots:
        print(f"  High-risk weeks ({len(hotspots)} found):")
        for h in hotspots:
            print(f"    {h['date']}  P={h['probability']:.4f}")
    else:
        print("  No high-risk weeks detected at threshold 0.85")
        print("  (Model needs more training data / epochs for reliable calibration)")

    print("\nDone.")


if __name__ == "__main__":
    main()
