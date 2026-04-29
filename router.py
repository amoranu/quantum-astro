"""
Phase 1 — Dynamic Astrological Router (Project Pitru-Maraka 2.0).

For v6 (angle-encoding), the router exposes `get_transit_features()` which
returns 13 continuous angles in radians:
   9 planet longitudes (Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn,
                        Rahu, Ketu) — degree precision via 2π/360°
   1 natal lagna sign  (0-11   → 2π/12)
   2 active dasha lords (MD, AD; cycling 0-8 → 2π/9)
   1 natal 8th house cusp longitude (degree precision)

The legacy `get_transit_encoding()` (4 nakshatra indices) is retained for
backward-compat with predict.py / pipeline.py callers that still want it.
"""

import math

import swisseph as swe
import pytz
from datetime import datetime

from dasha import compute_dasha

# Traditional Parashari sign-to-ruling-planet map (outer planets excluded).
# Index 0 = Aries, 1 = Taurus, ..., 11 = Pisces.
SIGN_RULERS: dict[int, str] = {
    0:  "Mars",      # Aries
    1:  "Venus",     # Taurus
    2:  "Mercury",   # Gemini
    3:  "Moon",      # Cancer
    4:  "Sun",       # Leo
    5:  "Mercury",   # Virgo
    6:  "Venus",     # Libra
    7:  "Mars",      # Scorpio  (traditional, not Pluto)
    8:  "Jupiter",   # Sagittarius
    9:  "Saturn",    # Capricorn
    10: "Saturn",    # Aquarius
    11: "Jupiter",   # Pisces
}

PLANET_SWE_IDS: dict[str, int] = {
    "Sun":     swe.SUN,
    "Moon":    swe.MOON,
    "Mars":    swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus":   swe.VENUS,
    "Saturn":  swe.SATURN,
    # Lunar nodes: Rahu = mean north node, Ketu = Rahu + 180° (computed below)
    "Rahu":    swe.MEAN_NODE,
    # "Ketu" is not a swisseph body — we derive it from Rahu in get_sidereal_longitude.
}

ALL_PLANETS_FOR_FEATURES: list[str] = [
    "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu",
]

NAKSHATRA_SIZE: float = 360.0 / 27.0   # 13.333...° per Nakshatra


# ---------------------------------------------------------------------------
# Low-level ephemeris helpers
# ---------------------------------------------------------------------------

def birth_to_jd(birth_date: str, birth_time: str, tz_str: str) -> float:
    """Convert birth data to Julian Day (UT)."""
    tz = pytz.timezone(tz_str)
    dt_local = datetime.strptime(f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M")
    try:
        dt_aware = tz.localize(dt_local, is_dst=None)
    except pytz.exceptions.AmbiguousTimeError:
        dt_aware = tz.localize(dt_local, is_dst=False)
    except pytz.exceptions.NonExistentTimeError:
        dt_aware = tz.localize(dt_local, is_dst=True)
    dt_utc = dt_aware.astimezone(pytz.utc)
    return swe.julday(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0,
    )


def date_to_jd(date_str: str) -> float:
    """Convert an ISO date string to Julian Day at noon UT."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return swe.julday(d.year, d.month, d.day, 12.0)


def get_sidereal_longitude(planet_name: str, jd: float) -> float:
    """Return Lahiri sidereal longitude (°) of *planet_name* at Julian Day *jd*.
    Ketu is derived as Rahu + 180°."""
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    if planet_name == "Ketu":
        rahu_lon = get_sidereal_longitude("Rahu", jd)
        return (rahu_lon + 180.0) % 360.0
    result, _ = swe.calc_ut(
        jd, PLANET_SWE_IDS[planet_name], swe.FLG_SIDEREAL | swe.FLG_SPEED
    )
    return result[0] % 360.0


def longitude_to_nakshatra(lon: float) -> int:
    """Map a sidereal longitude to a Nakshatra index in [0, 26]."""
    return int((lon % 360.0) / NAKSHATRA_SIZE) % 27


def _compute_lagna(birth_jd: float, lat: float, lon: float) -> tuple[float, int]:
    """
    Sidereal Ascendant via Placidus houses + Lahiri ayanamsa subtraction.
    Returns (longitude_degrees, sign_index 0-11).
    """
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    cusps, ascmc = swe.houses(birth_jd, lat, lon, b"P")
    ayanamsa = swe.get_ayanamsa(birth_jd)
    asc_sid = (ascmc[0] - ayanamsa) % 360.0
    return asc_sid, int(asc_sid / 30.0) % 12


# ---------------------------------------------------------------------------
# AstrologyRouter
# ---------------------------------------------------------------------------

class AstrologyRouter:
    """
    Maps the 4 abstract astrological roles to physical entities for one chart.

    Role → qubit group assignment:
        0  Universal Karaka   → Sun (hardcoded)         → qubits  0-4
        1  9th Lord           → ruler of 9th house sign → qubits  5-9
        2  8th Lord           → ruler of 8th house sign → qubits 10-14
                                (fallback to natal 8th cusp if ruler == 9th lord)
        3  Transiting Trigger → Saturn (hardcoded)      → qubits 15-19

    Overlap fallback:
        If 9th lord == 8th lord (only happens for Gemini Lagna where Saturn
        rules both Capricorn/8th and Aquarius/9th), the 8th Lord slot is
        replaced by the *natal* 8th house cusp longitude, encoded as its
        Nakshatra index.
    """

    def __init__(
        self,
        birth_date: str,
        birth_time: str,
        lat: float,
        lon: float,
        tz: str,
    ) -> None:
        self.birth_jd = birth_to_jd(birth_date, birth_time, tz)
        self.lat = lat
        self.lon = lon

        asc_lon, lagna_idx = _compute_lagna(self.birth_jd, lat, lon)
        self.lagna_sign_idx: int = lagna_idx
        self.asc_lon: float = asc_lon

        # House sign indices in [0, 11]
        self.ninth_sign_idx  = (lagna_idx + 8) % 12
        self.eighth_sign_idx = (lagna_idx + 7) % 12
        self.second_sign_idx = (lagna_idx + 1) % 12

        self.ninth_lord  = SIGN_RULERS[self.ninth_sign_idx]
        self._eighth_lord_planet = SIGN_RULERS[self.eighth_sign_idx]

        # Overlap check (Gemini lagna only: Saturn rules both 8th and 9th)
        self.eighth_uses_cusp: bool = (self.ninth_lord == self._eighth_lord_planet)
        self.eighth_cusp_lon: float = float(self.eighth_sign_idx * 30) % 360.0

        # Natal positions cached for chart-relative features
        self.natal_moon_lon:   float = get_sidereal_longitude("Moon",   self.birth_jd)
        self.natal_sun_lon:    float = get_sidereal_longitude("Sun",    self.birth_jd)
        self.natal_saturn_lon: float = get_sidereal_longitude("Saturn", self.birth_jd)

    # ------------------------------------------------------------------
    @property
    def eighth_lord(self) -> str:
        return "CUSP" if self.eighth_uses_cusp else self._eighth_lord_planet

    def get_roles(self) -> dict:
        """Human-readable summary of the 4 role assignments."""
        return {
            "karaka":           "Sun",
            "ninth_lord":       self.ninth_lord,
            "eighth_lord":      self.eighth_lord,
            "eighth_uses_cusp": self.eighth_uses_cusp,
            "eighth_cusp_lon":  self.eighth_cusp_lon if self.eighth_uses_cusp else None,
            "trigger":          "Saturn",
            "lagna_sign_idx":   self.lagna_sign_idx,
        }

    def get_transit_encoding(self, event_jd: float) -> list[int]:
        """
        Compute 4 Nakshatra indices (0-26) — one per abstract role — for the
        planetary positions on *event_jd*.

        Returns:
            [n_karaka, n_9th, n_8th, n_saturn]
        """
        # Role 0: Sun (hardcoded karaka of father)
        n0 = longitude_to_nakshatra(get_sidereal_longitude("Sun", event_jd))

        # Role 1: 9th Lord transit
        n1 = longitude_to_nakshatra(get_sidereal_longitude(self.ninth_lord, event_jd))

        # Role 2: 8th Lord transit, or static natal cusp nakshatra on fallback
        if self.eighth_uses_cusp:
            n2 = longitude_to_nakshatra(self.eighth_cusp_lon)
        else:
            n2 = longitude_to_nakshatra(
                get_sidereal_longitude(self._eighth_lord_planet, event_jd)
            )

        # Role 3: Saturn (hardcoded time-trigger)
        n3 = longitude_to_nakshatra(get_sidereal_longitude("Saturn", event_jd))

        return [n0, n1, n2, n3]

    # ------------------------------------------------------------------
    # v6 — angle features (continuous)
    # ------------------------------------------------------------------
    def get_transit_features(self, event_jd: float) -> list[float]:
        """
        Return 17 continuous features in radians (v11). All cyclic, [0, 2π].

        Indices 0-12 are the v9 features (chart-relative).
        Indices 13-16 are v11 additions (life-stage and natal-relative aspects).

        Ordered:
            [0-8]  9 planet longitudes RELATIVE TO NATAL LAGNA (mod 2π)
            [9]    Natal lagna sign idx (0-11) → 2π scaled
            [10]   Active Mahadasha lord (0-8) → 2π scaled
            [11]   Active Antardasha lord (0-8) → 2π scaled
            [12]   Saturn's longitude RELATIVE TO NATAL SUN
            [13]   Subject's age at event, normalised over 100y cycle
                   age_years / 100 × 2π — gives the model lifecycle context
            [14]   Saturn vs NATAL MOON (Sade Sati indicator)
                   primary Vedic timing tool for difficult / death events
            [15]   Saturn vs NATAL SATURN (Saturn return, ~29.5y cycle)
                   major life-transition indicator
            [16]   Mars vs NATAL SUN (Mars-Sun aspect strength proxy)
                   captures one Vedic aspect axis the model otherwise lacks
        """
        features: list[float] = []
        deg_to_rad = 2.0 * math.pi / 360.0
        asc_deg    = self.asc_lon

        # 0-8: planets, lagna-relative
        for planet in ALL_PLANETS_FOR_FEATURES:
            lon = get_sidereal_longitude(planet, event_jd)
            rel = (lon - asc_deg) % 360.0
            features.append(rel * deg_to_rad)

        # 9: lagna sign anchor
        features.append(self.lagna_sign_idx * (2.0 * math.pi / 12.0))

        # 10-11: dashas
        md_idx, ad_idx = compute_dasha(self.birth_jd, event_jd, self.natal_moon_lon)
        features.append(md_idx * (2.0 * math.pi / 9.0))
        features.append(ad_idx * (2.0 * math.pi / 9.0))

        # 12: Saturn vs natal Sun
        sat_lon = get_sidereal_longitude("Saturn", event_jd)
        features.append(((sat_lon - self.natal_sun_lon) % 360.0) * deg_to_rad)

        # 13: subject age at event (normalised over 100y cycle)
        age_years = (event_jd - self.birth_jd) / 365.25
        features.append((age_years / 100.0) * (2.0 * math.pi))

        # 14: Saturn vs natal Moon (Sade Sati)
        features.append(((sat_lon - self.natal_moon_lon) % 360.0) * deg_to_rad)

        # 15: Saturn vs natal Saturn (Saturn return)
        features.append(((sat_lon - self.natal_saturn_lon) % 360.0) * deg_to_rad)

        # 16: Mars vs natal Sun (aspect proxy)
        mars_lon = get_sidereal_longitude("Mars", event_jd)
        features.append(((mars_lon - self.natal_sun_lon) % 360.0) * deg_to_rad)

        return features
