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

DEFAULT_DATA = _HERE / "data" / "candidates_post1970.json"

DEFAULT_MODEL = _HERE / "model.pt"


# ---------------------------------------------------------------------------
# Month-pinpointing eval — per-subject argmax over the 5-year window
# ---------------------------------------------------------------------------

def _month_pinpoint_eval(model, X_test, test_subj: list, test_off: list) -> None:
    """For each test subject, find the month with highest predicted P(death)
    in their 5-year window. Report MAE in months between argmax and actual
    death (offset 0)."""
    import torch
    from collections import defaultdict

    print("\n" + "=" * 60, flush=True)
    print("  Month-pinpointing evaluation (test subjects only)", flush=True)
    print("=" * 60, flush=True)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).cpu().numpy()

    by_subject: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, (s, off) in enumerate(zip(test_subj, test_off)):
        by_subject[s].append((off, float(probs[i])))

    abs_errors: list[int] = []
    print(f"\n  {'subject':>8s}  {'argmax_off':>10s}  {'argmax_P':>9s}  {'|err|_mo':>9s}", flush=True)
    for s, pairs in by_subject.items():
        pairs.sort(key=lambda x: -x[1])  # by P descending
        arg_off, arg_p = pairs[0]
        err = abs(arg_off)
        abs_errors.append(err)
        print(f"  {s:>8d}  {arg_off:>+10d}  {arg_p:>9.4f}  {err:>9d}", flush=True)

    if abs_errors:
        sorted_e = sorted(abs_errors)
        n = len(sorted_e)
        median = sorted_e[n // 2]
        mean   = sum(sorted_e) / n
        within_1 = sum(1 for e in abs_errors if e <= 1) / n
        within_3 = sum(1 for e in abs_errors if e <= 3) / n
        within_6 = sum(1 for e in abs_errors if e <= 6) / n
        print(f"\n  N test subjects        : {n}", flush=True)
        print(f"  Mean |error| (months)  : {mean:.2f}", flush=True)
        print(f"  Median |error| (months): {median}", flush=True)
        print(f"  Within ±1 month        : {within_1*100:.1f}%", flush=True)
        print(f"  Within ±3 months       : {within_3*100:.1f}%", flush=True)
        print(f"  Within ±6 months       : {within_6*100:.1f}%", flush=True)
        print(f"  Random baseline MAE    : ~{30 / 2:.1f} months  (uniform argmax over 61 months)", flush=True)


# ---------------------------------------------------------------------------
# Smoke test — verify feature pipeline on one subject before launching a long run
# ---------------------------------------------------------------------------

def _run_smoke_test(data_path: Path) -> None:
    """Compute v6 features for one subject's death and one negative date, print
    all 13 angles + dasha lord names. Catches encoding/dasha bugs in seconds."""
    import math
    from datetime import datetime, timedelta
    from router import AstrologyRouter, date_to_jd
    from dasha import compute_dasha, VIMSHOTTARI_LORDS

    print("=" * 60, flush=True)
    print("  Feature smoke test", flush=True)
    print("=" * 60, flush=True)

    records = json.loads(data_path.read_text(encoding="utf-8"))
    valid = [r for r in records if r.get("father_death_date")]
    rec = valid[0]

    print(f"\nSubject: {rec.get('name', 'record-0')}", flush=True)
    print(f"  birth: {rec['birth_date']} {rec['birth_time']} @ ({rec['lat']}, {rec['lon']}) {rec['tz']}", flush=True)
    print(f"  father_death_date: {rec['father_death_date']}", flush=True)

    router = AstrologyRouter(
        rec["birth_date"], rec.get("birth_time") or "12:00",
        float(rec["lat"]), float(rec["lon"]), rec["tz"],
    )
    print(f"\nRouter:", flush=True)
    for k, v in router.get_roles().items():
        print(f"  {k:18s} = {v}", flush=True)
    print(f"  natal_moon_lon     = {router.natal_moon_lon:.3f}°", flush=True)

    feature_labels = [
        "Sun lon", "Moon lon", "Mars lon", "Mercury lon", "Jupiter lon",
        "Venus lon", "Saturn lon", "Rahu lon", "Ketu lon",
        "Lagna sign", "Mahadasha lord", "Antardasha lord", "8th cusp lon",
    ]

    death_jd = date_to_jd(rec["father_death_date"])
    neg_dt = datetime.strptime(rec["birth_date"], "%Y-%m-%d") + timedelta(days=15 * 365)
    neg_jd = date_to_jd(neg_dt.strftime("%Y-%m-%d"))

    for label, jd in [("DEATH DATE", death_jd), (f"NEG ({neg_dt.date()})", neg_jd)]:
        feats = router.get_transit_features(jd)
        md, ad = compute_dasha(router.birth_jd, jd, router.natal_moon_lon)
        print(f"\n--- {label} (JD={jd:.3f}) ---", flush=True)
        print(f"  Active Mahadasha = {VIMSHOTTARI_LORDS[md]}  Antardasha = {VIMSHOTTARI_LORDS[ad]}", flush=True)
        assert len(feats) == 13, f"Expected 13 features, got {len(feats)}"
        for lbl, f in zip(feature_labels, feats):
            deg = f * 180.0 / math.pi
            print(f"  {lbl:18s} = {f:+.4f} rad  ({deg:+7.2f}°)", flush=True)
        # Range check
        for lbl, f in zip(feature_labels, feats):
            assert 0.0 <= f <= 2.0 * math.pi + 1e-6, f"{lbl} out of [0, 2π]: {f}"

    print("\n[smoke] all 13 features in [0, 2π] ✓", flush=True)
    print("[smoke] dasha lookup ✓", flush=True)
    print("[smoke] feature-pipeline OK — safe to launch training\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Project Pitru-Maraka 2.0")
    parser.add_argument("--data",       default=str(DEFAULT_DATA), help="Path to candidates JSON")
    parser.add_argument("--model",      default=str(DEFAULT_MODEL), help="Path to save / load model")
    parser.add_argument("--skip-train", action="store_true",        help="Skip training, load existing model")
    parser.add_argument("--epochs",     type=int,   default=30,     help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=16,     help="Batch size")
    parser.add_argument("--lr",         type=float, default=0.01,   help="Initial Adam LR (cosine-annealed)")
    parser.add_argument("--test-frac",  type=float, default=0.4,    help="Subject-level test split fraction")
    parser.add_argument("--window-months", type=int, default=30,
                        help="±k months around death for v8 windowed sampling")
    parser.add_argument("--pos-months", type=int,   default=2,
                        help="±k months from death labelled as positive")
    parser.add_argument("--split-seed", type=int,   default=7,      help="Subject split RNG seed")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Permutation test: shuffle Y_train before training (real signal should fail)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run feature-extraction sanity check on 1 subject and exit (no training)")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(Path(args.data))
        return

    data_path  = Path(args.data)
    model_path = Path(args.model)

    print("=" * 60)
    print("  Project Pitru-Maraka 2.0 — Dynamic Quantum Astrology VQE")
    if args.shuffle_labels:
        print("  ⚠ PERMUTATION TEST MODE — Y_train shuffled")
    print("=" * 60)

    # ── Phase 2: Build dataset (subject-level split) ─────────────────────
    if not args.skip_train:
        print(f"\n[Phase 2] Building dataset from:\n  {data_path}\n")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        from pipeline import build_dataset_window
        X_tr, Y_tr, X_te, Y_te, train_subj, train_off, test_subj, test_off, info = build_dataset_window(
            data_path,
            test_frac=args.test_frac,
            window_months=args.window_months,
            pos_months=args.pos_months,
            split_seed=args.split_seed,
            shuffle_labels=args.shuffle_labels,
        )

        # ── Phases 3+4: Train ──────────────────────────────────────────────
        print("\n[Phases 3+4] Initialising 16-qubit VQC and training …\n")
        from train import train
        model = train(
            X_tr, Y_tr,
            X_test=X_te, Y_test=Y_te,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=model_path,
        )

        # ── Month-pinpointing eval: per-subject argmax over the 5-yr window ─
        _month_pinpoint_eval(model, X_te, test_subj, test_off)
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
    print(f"  Features  : [{', '.join(f'{a:.3f}' for a in result['features'])}]")
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
