"""
Phase 5 — Inference API (Project Pitru-Maraka 2.0).

predict(birth_data, target_date, model) → dict

    birth_data : {birth_date, birth_time, lat, lon, tz}
    target_date: ISO date string to evaluate (YYYY-MM-DD)
    model      : trained QMLModel (from train.load_model)

Returns a dict with:
    prediction  : "High Risk Window" if P(death) ≥ 0.85, else "Baseline"
    probability : float in [0, 1]
    roles       : dynamic planet assignments for this subject
    nakshatras  : [n_sun, n_9th, n_8th, n_saturn]
    target_date : echo of the input date
"""

from __future__ import annotations

from pathlib import Path

import torch

from router import AstrologyRouter, date_to_jd
from pipeline import encode_four_nakshatras
from train import QMLModel, load_model

RISK_THRESHOLD = 0.85


def predict(
    birth_data: dict,
    target_date: str,
    model: QMLModel,
) -> dict:
    """
    Evaluate the father-loss risk for *target_date* given a birth chart.

    Args:
        birth_data  : dict with keys birth_date, birth_time, lat, lon, tz.
        target_date : ISO date string (YYYY-MM-DD).
        model       : locked, trained QMLModel (eval mode, no gradients).

    Returns:
        Prediction dict (see module docstring).
    """
    # 5.1 — Route planets for this subject
    router = AstrologyRouter(
        birth_data["birth_date"],
        birth_data.get("birth_time") or "12:00",
        float(birth_data["lat"]),
        float(birth_data["lon"]),
        birth_data["tz"],
    )
    roles = router.get_roles()

    # 5.2 — Encode the transit sky on target_date
    event_jd   = date_to_jd(target_date)
    nakshatras = router.get_transit_encoding(event_jd)
    bits       = encode_four_nakshatras(nakshatras)
    x          = torch.tensor([bits], dtype=torch.float32)   # (1, 20)

    # 5.3 — Forward pass (inference only) — model returns logits, apply σ
    model.eval()
    with torch.no_grad():
        prob: float = torch.sigmoid(model(x)).item()

    # 5.4 — Decision gate
    if prob >= RISK_THRESHOLD:
        prediction = "High Risk Window"
    else:
        prediction = "Baseline"

    return {
        "prediction":  prediction,
        "probability": round(prob, 6),
        "roles":       roles,
        "nakshatras":  nakshatras,
        "target_date": target_date,
    }


def scan_window(
    birth_data: dict,
    start_date: str,
    end_date: str,
    model: QMLModel,
    step_days: int = 7,
) -> list[dict]:
    """
    Slide a weekly window over [start_date, end_date] and collect all dates
    whose P(death) ≥ RISK_THRESHOLD.

    Useful for identifying clusters of high-risk epochs in a time range.
    """
    from datetime import datetime, timedelta

    results: list[dict] = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt  = datetime.strptime(end_date,   "%Y-%m-%d")

    router = AstrologyRouter(
        birth_data["birth_date"],
        birth_data.get("birth_time") or "12:00",
        float(birth_data["lat"]),
        float(birth_data["lon"]),
        birth_data["tz"],
    )

    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        try:
            event_jd   = date_to_jd(date_str)
            nakshatras = router.get_transit_encoding(event_jd)
            bits       = encode_four_nakshatras(nakshatras)
            x          = torch.tensor([bits], dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()

            if prob >= RISK_THRESHOLD:
                results.append({"date": date_str, "probability": round(prob, 6)})
        except Exception:
            pass

        current += timedelta(days=step_days)

    return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        print("Run main.py first to train the model.")
        sys.exit(1)

    m = load_model(model_path)

    # Demo: Dale Earnhardt Jr. — father died 2001-02-18
    demo = {
        "birth_date": "1974-10-10",
        "birth_time": "05:33",
        "lat":  35.4167,
        "lon":  -80.5833,
        "tz":   "America/New_York",
    }
    result = predict(demo, "2001-02-18", m)
    print(json.dumps(result, indent=2, default=str))
