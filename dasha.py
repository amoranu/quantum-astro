"""
Vimshottari Mahadasha + Antardasha computation (Project Pitru-Maraka 2.0).

Vimshottari is the principal Vedic dasha system. The 120-year cycle distributes
unevenly across 9 lords; the starting lord is determined by the nakshatra of
the Moon at birth, and the elapsed fraction within that nakshatra sets how
much of the first MD has already passed.

Why dasha matters here: in Jyotish, transits "fire" through the active dasha
lord. A malefic transit during a benefic dasha is muted; a moderate transit
during a death-indicator's dasha can become decisive. Pure-transit models
miss this gating completely.
"""

from __future__ import annotations

# ─── Vimshottari order: (lord, period_years) ──────────────────────────────────
VIMSHOTTARI_ORDER: list[tuple[str, int]] = [
    ("Ketu",     7),
    ("Venus",   20),
    ("Sun",      6),
    ("Moon",    10),
    ("Mars",     7),
    ("Rahu",    18),
    ("Jupiter", 16),
    ("Saturn",  19),
    ("Mercury", 17),
]
VIMSHOTTARI_TOTAL = 120  # years (sum of all periods)

VIMSHOTTARI_LORDS: list[str] = [p[0] for p in VIMSHOTTARI_ORDER]
PLANET_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(VIMSHOTTARI_LORDS)}

# Each of the 27 nakshatras is "owned" by a Vimshottari lord, cycling through
# the 9 lords three times: Ashwini=Ketu, Bharani=Venus, Krittika=Sun, …, Revati=Mercury.
NAKSHATRA_LORDS: list[str] = VIMSHOTTARI_LORDS * 3   # length 27

NAKSHATRA_SIZE = 360.0 / 27.0    # 13.333…° per nakshatra
JULIAN_YEAR    = 365.25          # days


def compute_dasha(
    birth_jd: float,
    event_jd: float,
    moon_lon_birth: float,
) -> tuple[int, int]:
    """
    Returns (md_lord_idx, ad_lord_idx) for the dasha active at *event_jd*.

    Indices are into VIMSHOTTARI_LORDS, i.e. 0=Ketu … 8=Mercury (the dasha
    cyclic order, NOT the natural-zodiac order).

    Args:
        birth_jd       : Julian Day (UT) of birth.
        event_jd       : Julian Day (UT) of the event.
        moon_lon_birth : Moon's sidereal longitude (°) at birth.
    """
    # ── 1. Starting MD lord and elapsed fraction ───────────────────────────
    moon_nak       = int((moon_lon_birth % 360.0) / NAKSHATRA_SIZE) % 27
    pos_in_nak     = (moon_lon_birth % NAKSHATRA_SIZE) / NAKSHATRA_SIZE  # 0..1
    first_lord     = NAKSHATRA_LORDS[moon_nak]
    first_idx      = PLANET_TO_IDX[first_lord]
    first_period_y = VIMSHOTTARI_ORDER[first_idx][1]

    # Years already elapsed in the starting MD at the moment of birth.
    elapsed_in_first = pos_in_nak * first_period_y

    # ── 2. Years from birth to event, plus the pre-elapsed first-MD time ──
    years_to_event = (event_jd - birth_jd) / JULIAN_YEAR + elapsed_in_first

    # ── 3. Walk Vimshottari forward to find the active MD ─────────────────
    cum_years = 0.0
    md_idx    = first_idx
    md_period = first_period_y
    time_in_md = 0.0
    for _ in range(VIMSHOTTARI_TOTAL):  # safety bound (one full cycle)
        period = VIMSHOTTARI_ORDER[md_idx][1]
        if cum_years + period > years_to_event:
            md_period  = period
            time_in_md = years_to_event - cum_years
            break
        cum_years += period
        md_idx = (md_idx + 1) % 9
    else:
        # Beyond 120 years — clamp to last MD
        md_idx     = (first_idx - 1) % 9
        md_period  = VIMSHOTTARI_ORDER[md_idx][1]
        time_in_md = md_period - 1e-6

    # ── 4. Within current MD, find active AD ──────────────────────────────
    # Antardashas start with the same lord as the MD, then proceed in
    # Vimshottari order. AD duration = MD_period × AD_lord_period / 120.
    cum_ad = 0.0
    ad_idx = md_idx
    for _ in range(9):
        ad_lord_period = VIMSHOTTARI_ORDER[ad_idx][1]
        ad_duration_y  = (md_period * ad_lord_period) / VIMSHOTTARI_TOTAL
        if cum_ad + ad_duration_y > time_in_md:
            return md_idx, ad_idx
        cum_ad += ad_duration_y
        ad_idx = (ad_idx + 1) % 9

    # Fallthrough — return last AD if floating-point sliced us past the end.
    return md_idx, ad_idx
