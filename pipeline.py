"""
Phase 2 — Data Pipeline & Encoding (Project Pitru-Maraka 2.0).

Loads candidates_post1970.json, routes each chart through AstrologyRouter,
converts 5 Nakshatra indices to a flat 25-bit binary array, and assembles
the (X, Y) training tensors.

Positive samples : the actual father_death_date for each subject (label=1).
Negative samples : random dates sampled from [birth+10yr, death-180d] when
                   the father was verifiably alive (label=0).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch

from router import AstrologyRouter, date_to_jd


# ---------------------------------------------------------------------------
# Bit-encoding helpers
# ---------------------------------------------------------------------------

def nakshatra_to_bits(idx: int) -> list[int]:
    """Nakshatra index 0-26 → 5 MSB-first bits (2^5=32 > 27)."""
    return [(idx >> (4 - b)) & 1 for b in range(5)]


def encode_five_nakshatras(indices: list[int]) -> list[int]:
    """5 Nakshatra indices → flat 25-bit list."""
    bits: list[int] = []
    for idx in indices:
        bits.extend(nakshatra_to_bits(idx))
    return bits


# ---------------------------------------------------------------------------
# Negative-date sampler
# ---------------------------------------------------------------------------

def sample_negative_dates(
    birth_date: str,
    father_death_date: str,
    n: int,
    seed: int = 0,
) -> list[str]:
    """
    Return up to *n* ISO date strings in [birth+10yr, death-180d].
    No date is within 180 days of father_death_date.
    """
    rng = random.Random(seed)
    birth_dt = datetime.strptime(birth_date, "%Y-%m-%d")
    death_dt = datetime.strptime(father_death_date, "%Y-%m-%d")

    start = birth_dt + timedelta(days=10 * 365)
    end   = death_dt - timedelta(days=180)

    if start >= end:
        start = birth_dt + timedelta(days=5 * 365)
        end   = death_dt - timedelta(days=30)

    if start >= end:
        return []

    total_days = (end - start).days
    seen: set[str] = set()
    dates: list[str] = []

    for _ in range(n * 5):                     # oversample to fill quota
        offset = rng.randint(0, total_days)
        d_str = (start + timedelta(days=offset)).strftime("%Y-%m-%d")
        if d_str not in seen:
            seen.add(d_str)
            dates.append(d_str)
        if len(dates) >= n:
            break

    return dates


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    json_path: str | Path,
    target_size: int = 1000,
    neg_seed: int = 42,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (X, Y) tensors from candidates_post1970.json.

    Args:
        json_path    : path to the JSON file.
        target_size  : approximate total number of training samples.
        neg_seed     : base RNG seed for negative-date sampling.
        verbose      : print progress.

    Returns:
        X : FloatTensor (N, 25)  — binary transit encodings.
        Y : FloatTensor (N, 1)   — labels (1 = death event, 0 = baseline).
    """
    records: list[dict] = json.loads(Path(json_path).read_text(encoding="utf-8"))

    valid = [r for r in records if r.get("father_death_date")]
    if verbose:
        print(
            f"[pipeline] {len(valid)} records with father_death_date "
            f"(out of {len(records)} total)"
        )

    n_pos = len(valid)
    n_neg_per = max(2, (target_size - n_pos) // max(n_pos, 1))

    rows_X: list[list[int]] = []
    rows_Y: list[int]       = []
    n_skipped = 0

    for i, rec in enumerate(valid):
        name         = rec.get("name", f"record-{i}")
        birth_date   = rec["birth_date"]
        birth_time   = rec.get("birth_time") or "12:00"
        lat          = float(rec["lat"])
        lon          = float(rec["lon"])
        tz           = rec["tz"]
        death_date   = rec["father_death_date"]

        # Initialise router for this subject
        try:
            router = AstrologyRouter(birth_date, birth_time, lat, lon, tz)
        except Exception as exc:
            if verbose:
                print(f"  [skip] {name}: router failed — {exc}")
            n_skipped += 1
            continue

        # Positive sample: transits on the actual death date
        try:
            death_jd = date_to_jd(death_date)
            naks = router.get_transit_encoding(death_jd)
            rows_X.append(encode_five_nakshatras(naks))
            rows_Y.append(1)
        except Exception as exc:
            if verbose:
                print(f"  [skip] {name}: death-date encoding failed — {exc}")
            n_skipped += 1
            continue

        # Negative samples: random "alive" dates
        neg_dates = sample_negative_dates(
            birth_date, death_date, n_neg_per, seed=neg_seed + i * 1000
        )
        for neg_date in neg_dates:
            try:
                neg_jd = date_to_jd(neg_date)
                naks_neg = router.get_transit_encoding(neg_jd)
                rows_X.append(encode_five_nakshatras(naks_neg))
                rows_Y.append(0)
            except Exception:
                continue

        if verbose and (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{n_pos} subjects …")

    n_pos_actual = sum(rows_Y)
    n_neg_actual = len(rows_Y) - n_pos_actual
    if verbose:
        print(
            f"[pipeline] dataset: {len(rows_X)} samples  "
            f"({n_pos_actual} positive | {n_neg_actual} negative | "
            f"{n_skipped} subjects skipped)"
        )

    X = torch.tensor(rows_X, dtype=torch.float32)
    Y = torch.tensor(rows_Y, dtype=torch.float32).unsqueeze(1)
    return X, Y


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        Path(__file__).parent.parent
        / "neuro-symbolic-astro"
        / "astroql" / "applications" / "father_longevity"
        / "data" / "candidates_post1970.json"
    )
    X, Y = build_dataset(path)
    print(f"X shape: {X.shape}  Y shape: {Y.shape}")
    print(f"Class balance: {Y.mean().item():.3f} (fraction positive)")
