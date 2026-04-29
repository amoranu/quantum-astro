"""
Phase 2 — Data Pipeline & Encoding (Project Pitru-Maraka 2.0).

v6: 13-dim continuous angle features (radians) per sample. Replaces the
v5 4-nakshatra binary encoding (20 bits) with full degree-precision
longitudes for 9 planets + lagna + Mahadasha + Antardasha + 8th cusp.

Subject-level split is critical: row-level random splits would leak each
subject's transit signature across train and test (positive day shares
slow-moving planets with neighboring negative days), inflating accuracy.

Positive samples : the death date ± k days for k ∈ {-AUG_DAYS..+AUG_DAYS}.
                    73 unique subjects × (2·AUG_DAYS+1) days = 511 positives
                    when AUG_DAYS=3. Augmentation reflects the actual
                    astrological semantics — slow planets (Saturn) move
                    ~1°/30d so death_date±3d shares the same transit pattern.
Negative samples : random dates from [birth+10y, death-NEG_BUFFER_D] when
                    the father was verifiably alive. Buffer prevents
                    augmented positives from contradicting nearby negatives.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import torch

from router import AstrologyRouter, date_to_jd

AUG_DAYS = 3        # legacy: positive augmentation half-window (k ∈ {-3..+3})
NEG_BUFFER_D = 30   # legacy: min gap between any negative date and death_date


def _add_months(dt: datetime, months: int) -> datetime:
    """Add `months` calendar months to dt, snapping to day 15 to avoid month-end clipping."""
    total = dt.month - 1 + months
    new_year = dt.year + total // 12
    new_month = (total % 12) + 1
    return dt.replace(year=new_year, month=new_month, day=15)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
# v6 uses continuous angle features straight from router.get_transit_features().
# The legacy nakshatra binary helpers below are retained for backward compat
# with predict.py (used by older saved models).

def nakshatra_to_bits(idx: int) -> list[int]:
    """Nakshatra index 0-26 → 5 MSB-first bits (2^5=32 > 27). Legacy."""
    return [(idx >> (4 - b)) & 1 for b in range(5)]


def encode_four_nakshatras(indices: list[int]) -> list[int]:
    """4 Nakshatra indices → flat 20-bit list. Legacy."""
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
    neg_window_y: int = 2,
) -> list[str]:
    """
    Return up to *n* ISO date strings in an *age-matched* window:
    [death - neg_window_y, death - NEG_BUFFER_D].

    Why age-matched: previously negatives were sampled from [birth+10y, death-30d],
    a wide range systematically *younger* than the death event. The model could
    learn a trivial age/era confounder ("subjects in their 50s during the 1990s
    have these transit patterns") rather than a death-specific signature.
    Sampling negatives within 2 years of the death event removes that shortcut —
    the subject is alive in those days, but the natal/dasha context is identical
    and slow-planet positions are very close.
    """
    rng = random.Random(seed)
    birth_dt = datetime.strptime(birth_date, "%Y-%m-%d")
    death_dt = datetime.strptime(father_death_date, "%Y-%m-%d")

    start = death_dt - timedelta(days=neg_window_y * 365)
    # Floor: subject must be at least 10 years old in the negative window.
    min_age_start = birth_dt + timedelta(days=10 * 365)
    if start < min_age_start:
        start = min_age_start
    end = death_dt - timedelta(days=NEG_BUFFER_D)

    if start >= end:
        # Edge case: subject's father died young; fall back to anything older than 5y.
        start = birth_dt + timedelta(days=5 * 365)
        end   = death_dt - timedelta(days=NEG_BUFFER_D)

    if start >= end:
        return []

    total_days = (end - start).days
    seen: set[str] = set()
    dates: list[str] = []

    for _ in range(n * 5):
        offset = rng.randint(0, total_days)
        d_str = (start + timedelta(days=offset)).strftime("%Y-%m-%d")
        if d_str not in seen:
            seen.add(d_str)
            dates.append(d_str)
        if len(dates) >= n:
            break

    return dates


# ---------------------------------------------------------------------------
# Per-subject sample assembly
# ---------------------------------------------------------------------------

def _build_subject_samples(
    rec: dict,
    n_neg_per: int,
    aug_days: int,
    neg_seed: int,
) -> tuple[list[list[float]], list[int]] | None:
    """
    Build (X_rows, Y_rows) for a single subject using v6 angle features.

    X_rows[i] is a list of 13 floats (angles in radians).
    Returns None if the subject's chart cannot be routed.
    """
    birth_date = rec["birth_date"]
    birth_time = rec.get("birth_time") or "12:00"
    lat        = float(rec["lat"])
    lon        = float(rec["lon"])
    tz         = rec["tz"]
    death_date = rec["father_death_date"]

    try:
        router = AstrologyRouter(birth_date, birth_time, lat, lon, tz)
    except Exception:
        return None

    X_rows: list[list[float]] = []
    Y_rows: list[int]         = []

    death_dt = datetime.strptime(death_date, "%Y-%m-%d")
    for k in range(-aug_days, aug_days + 1):
        try:
            d  = death_dt + timedelta(days=k)
            jd = date_to_jd(d.strftime("%Y-%m-%d"))
            X_rows.append(router.get_transit_features(jd))
            Y_rows.append(1)
        except Exception:
            continue

    if not Y_rows:
        return None

    neg_dates = sample_negative_dates(birth_date, death_date, n_neg_per, seed=neg_seed)
    for neg_date in neg_dates:
        try:
            jd = date_to_jd(neg_date)
            X_rows.append(router.get_transit_features(jd))
            Y_rows.append(0)
        except Exception:
            continue

    return X_rows, Y_rows


# ---------------------------------------------------------------------------
# Train/test split builder
# ---------------------------------------------------------------------------

def build_dataset_split(
    json_path: str | Path,
    test_frac: float = 0.4,
    n_neg_per: int = 12,
    aug_days: int = AUG_DAYS,
    split_seed: int = 7,
    neg_seed_base: int = 42,
    shuffle_labels: bool = False,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Subject-level train/test split with positive augmentation.

    Args:
        json_path       : path to candidates JSON.
        test_frac       : fraction of subjects (not rows) reserved for test.
        n_neg_per       : negative dates per subject.
        aug_days        : positive ±k days; total positives per subject = 2k+1.
        split_seed      : RNG seed for the subject partition.
        neg_seed_base   : base seed for negative-date sampling (per-subject offset).
        shuffle_labels  : if True, shuffle Y_train BEFORE returning. Used for the
                          permutation test — a real signal should not survive it.
        verbose         : print progress.

    Returns:
        X_train, Y_train, X_test, Y_test, info
            X_*: FloatTensor (N, 20); Y_*: FloatTensor (N, 1)
            info: dict with split metadata for logging.
    """
    records: list[dict] = json.loads(Path(json_path).read_text(encoding="utf-8"))
    valid = [r for r in records if r.get("father_death_date")]

    if verbose:
        print(
            f"[pipeline] {len(valid)} subjects | aug ±{aug_days}d "
            f"| neg/subject={n_neg_per} | buffer={NEG_BUFFER_D}d "
            f"| test_frac={test_frac} | split_seed={split_seed}",
            flush=True,
        )

    rng = random.Random(split_seed)
    indices = list(range(len(valid)))
    rng.shuffle(indices)
    n_test = int(round(len(valid) * test_frac))
    test_idx  = set(indices[:n_test])
    train_idx = set(indices[n_test:])

    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    n_skipped = 0
    n_subj_train = n_subj_test = 0

    for i, rec in enumerate(valid):
        result = _build_subject_samples(rec, n_neg_per, aug_days, neg_seed_base + i * 1000)
        if result is None:
            n_skipped += 1
            continue
        X_rows, Y_rows = result
        if i in train_idx:
            X_tr.extend(X_rows); Y_tr.extend(Y_rows); n_subj_train += 1
        else:
            X_te.extend(X_rows); Y_te.extend(Y_rows); n_subj_test += 1

        if verbose and (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(valid)} subjects …", flush=True)

    X_train = torch.tensor(X_tr, dtype=torch.float32)
    Y_train = torch.tensor(Y_tr, dtype=torch.float32).unsqueeze(1)
    X_test  = torch.tensor(X_te, dtype=torch.float32)
    Y_test  = torch.tensor(Y_te, dtype=torch.float32).unsqueeze(1)

    if shuffle_labels:
        # Permutation test — real signal should drop accuracy to ~baseline.
        perm = torch.randperm(Y_train.shape[0])
        Y_train = Y_train[perm]
        if verbose:
            print("[pipeline] !! shuffle_labels=True — Y_train permuted (permutation test mode)", flush=True)

    info = {
        "n_subjects_total": len(valid),
        "n_subjects_train": n_subj_train,
        "n_subjects_test":  n_subj_test,
        "n_skipped":        n_skipped,
        "train_pos":        int(Y_train.sum().item()),
        "train_neg":        int(Y_train.shape[0] - Y_train.sum().item()),
        "test_pos":         int(Y_test.sum().item()),
        "test_neg":         int(Y_test.shape[0] - Y_test.sum().item()),
        "aug_days":         aug_days,
        "neg_buffer_d":     NEG_BUFFER_D,
        "shuffle_labels":   shuffle_labels,
    }

    if verbose:
        print(
            f"[pipeline] split done — train: {n_subj_train} subj | "
            f"{info['train_pos']} pos / {info['train_neg']} neg "
            f"({info['train_pos']/(info['train_pos']+info['train_neg'])*100:.1f}% pos)",
            flush=True,
        )
        print(
            f"[pipeline] split done — test : {n_subj_test} subj | "
            f"{info['test_pos']} pos / {info['test_neg']} neg "
            f"({info['test_pos']/(info['test_pos']+info['test_neg'])*100:.1f}% pos)",
            flush=True,
        )
        if n_skipped:
            print(f"[pipeline] {n_skipped} subjects skipped (router/encoding failure)", flush=True)

    return X_train, Y_train, X_test, Y_test, info


# ---------------------------------------------------------------------------
# v8 windowed builder — month-precision around death event
# ---------------------------------------------------------------------------

def _build_subject_window_samples(
    rec: dict,
    window_months: int,
    pos_months: int,
) -> tuple[list[list[float]], list[int], list[int]] | None:
    """
    Generate (window_months × 2 + 1) monthly samples for one subject.

    Returns (X_rows, Y_rows, offsets) where:
      X_rows[i]  = 13 angle features (radians)
      Y_rows[i]  = 1 if |month_offset| ≤ pos_months else 0
      offsets[i] = signed month offset from death (-window_months..+window_months)
    or None if router init fails.
    """
    birth_date = rec["birth_date"]
    birth_time = rec.get("birth_time") or "12:00"
    lat        = float(rec["lat"])
    lon        = float(rec["lon"])
    tz         = rec["tz"]
    death_date = rec["father_death_date"]

    try:
        router = AstrologyRouter(birth_date, birth_time, lat, lon, tz)
    except Exception:
        return None

    death_dt = datetime.strptime(death_date, "%Y-%m-%d")

    X_rows: list[list[float]] = []
    Y_rows: list[int]         = []
    offsets: list[int]        = []

    for k in range(-window_months, window_months + 1):
        try:
            sample_dt = _add_months(death_dt, k)
            jd = date_to_jd(sample_dt.strftime("%Y-%m-%d"))
            X_rows.append(router.get_transit_features(jd))
            Y_rows.append(1 if abs(k) <= pos_months else 0)
            offsets.append(k)
        except Exception:
            continue

    if not Y_rows or all(y == 0 for y in Y_rows):
        return None
    return X_rows, Y_rows, offsets


def build_dataset_window(
    json_path: str | Path,
    test_frac: float = 0.4,
    window_months: int = 30,        # ±30 months = 5-year window (61 monthly samples)
    pos_months: int = 2,            # death ±2 months → 5 positive months per subject
    split_seed: int = 7,
    shuffle_labels: bool = False,
    verbose: bool = True,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    list[int], list[int], list[int], list[int], dict,
]:
    """
    Subject-level 60/40 split with monthly-resolution windowed sampling.

    For each subject: 2·window_months+1 samples = 1 sample per month over a
    (2·window_months/12)-year window centred on death_date. Labels are 1 within
    ±pos_months of death, else 0. Slow planets barely move month-to-month, so
    discrimination must come from fast planet/dasha shifts.

    Returns:
        X_train, Y_train, X_test, Y_test  : tensors
        train_subj, train_offsets         : per-row metadata for train split
        test_subj,  test_offsets          : per-row metadata for test split
                                            (used by month-pinpointing eval)
        info                              : split summary dict
    """
    records: list[dict] = json.loads(Path(json_path).read_text(encoding="utf-8"))
    valid = [r for r in records if r.get("father_death_date")]

    if verbose:
        print(
            f"[pipeline] {len(valid)} subjects | window=±{window_months}mo "
            f"| pos=±{pos_months}mo | test_frac={test_frac} | split_seed={split_seed}",
            flush=True,
        )

    rng = random.Random(split_seed)
    indices = list(range(len(valid)))
    rng.shuffle(indices)
    n_test = int(round(len(valid) * test_frac))
    test_idx_set  = set(indices[:n_test])

    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    train_subj, train_offsets = [], []
    test_subj,  test_offsets  = [], []
    n_skipped = n_subj_train = n_subj_test = 0

    for i, rec in enumerate(valid):
        result = _build_subject_window_samples(rec, window_months, pos_months)
        if result is None:
            n_skipped += 1
            continue
        X_rows, Y_rows, offsets = result
        if i in test_idx_set:
            X_te.extend(X_rows); Y_te.extend(Y_rows)
            test_subj.extend([i] * len(X_rows))
            test_offsets.extend(offsets)
            n_subj_test += 1
        else:
            X_tr.extend(X_rows); Y_tr.extend(Y_rows)
            train_subj.extend([i] * len(X_rows))
            train_offsets.extend(offsets)
            n_subj_train += 1

        if verbose and (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(valid)} subjects …", flush=True)

    X_train = torch.tensor(X_tr, dtype=torch.float32)
    Y_train = torch.tensor(Y_tr, dtype=torch.float32).unsqueeze(1)
    X_test  = torch.tensor(X_te, dtype=torch.float32)
    Y_test  = torch.tensor(Y_te, dtype=torch.float32).unsqueeze(1)

    if shuffle_labels:
        perm = torch.randperm(Y_train.shape[0])
        Y_train = Y_train[perm]
        if verbose:
            print("[pipeline] !! shuffle_labels=True — Y_train permuted", flush=True)

    info = {
        "n_subjects_total": len(valid),
        "n_subjects_train": n_subj_train,
        "n_subjects_test":  n_subj_test,
        "n_skipped":        n_skipped,
        "train_pos":        int(Y_train.sum().item()),
        "train_neg":        int(Y_train.shape[0] - Y_train.sum().item()),
        "test_pos":         int(Y_test.sum().item()),
        "test_neg":         int(Y_test.shape[0] - Y_test.sum().item()),
        "window_months":    window_months,
        "pos_months":       pos_months,
        "shuffle_labels":   shuffle_labels,
    }

    if verbose:
        print(
            f"[pipeline] split done — train: {n_subj_train} subj | "
            f"{info['train_pos']} pos / {info['train_neg']} neg "
            f"({info['train_pos']/(info['train_pos']+info['train_neg'])*100:.1f}% pos)",
            flush=True,
        )
        print(
            f"[pipeline] split done — test : {n_subj_test} subj | "
            f"{info['test_pos']} pos / {info['test_neg']} neg "
            f"({info['test_pos']/(info['test_pos']+info['test_neg'])*100:.1f}% pos)",
            flush=True,
        )
        if n_skipped:
            print(f"[pipeline] {n_skipped} subjects skipped (router/encoding failure)", flush=True)

    return X_train, Y_train, X_test, Y_test, train_subj, train_offsets, test_subj, test_offsets, info


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        Path(__file__).parent / "data" / "candidates_post1970.json"
    )
    X_tr, Y_tr, X_te, Y_te, info = build_dataset_split(path)
    print(f"X_train: {X_tr.shape}  Y_train: {Y_tr.shape}")
    print(f"X_test : {X_te.shape}  Y_test : {Y_te.shape}")
    print(f"info: {info}")
