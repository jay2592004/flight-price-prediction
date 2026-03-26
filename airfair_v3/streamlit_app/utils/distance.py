"""
utils/distance.py
Haversine distance calculator + full city coordinate map.
Used by the predictor so users can calculate distance directly
from city pairs instead of entering it manually.
"""

import math
from typing import Optional, Tuple

# ── City coordinates (lat, lon) ───────────────────────────────────────────────
CITY_COORDS: dict[str, Tuple[float, float]] = {
    "Ahmedabad":  (23.0225,  72.5714),
    "Bangalore":  (12.9716,  77.5946),
    "Bangkok":    (13.7563, 100.5018),
    "Chennai":    (13.0827,  80.2707),
    "Delhi":      (28.6139,  77.2090),
    "Doha":       (25.2854,  51.5310),
    "Dubai":      (25.2048,  55.2708),
    "Frankfurt":  (50.1109,   8.6821),
    "Hong Kong":  (22.3193, 114.1694),
    "Istanbul":   (41.0082,  28.9784),
    "London":     (51.5074,  -0.1278),
    "Mumbai":     (19.0760,  72.8777),
    "New York":   (40.7128, -74.0060),
    "Paris":      (48.8566,   2.3522),
    "Singapore":  (1.3521,  103.8198),
}

# ── Full distance lookup (exact same as generator) ────────────────────────────
DIST_LOOKUP: dict[Tuple[str, str], int] = {
    ("Delhi","Mumbai"):1400,("Delhi","Bangalore"):2150,("Delhi","Chennai"):2200,
    ("Delhi","Ahmedabad"):950,("Delhi","London"):6700,("Delhi","Dubai"):2200,
    ("Delhi","Singapore"):4150,("Delhi","Frankfurt"):6200,("Delhi","New York"):11750,
    ("Delhi","Bangkok"):2900,("Delhi","Hong Kong"):3750,("Delhi","Paris"):6600,
    ("Delhi","Istanbul"):4200,("Delhi","Doha"):2350,
    ("Mumbai","Bangalore"):980,("Mumbai","Chennai"):1330,("Mumbai","Ahmedabad"):530,
    ("Mumbai","London"):7200,("Mumbai","Dubai"):1950,("Mumbai","Singapore"):4180,
    ("Mumbai","Frankfurt"):7050,("Mumbai","New York"):12550,("Mumbai","Bangkok"):3080,
    ("Mumbai","Hong Kong"):4250,("Mumbai","Paris"):6950,("Mumbai","Istanbul"):4400,
    ("Mumbai","Doha"):1960,("London","New York"):5540,("London","Dubai"):5500,
    ("London","Singapore"):10850,("London","Hong Kong"):9600,("London","Frankfurt"):650,
    ("London","Paris"):340,("London","Istanbul"):2510,("London","Bangkok"):9540,
    ("London","Doha"):5500,("Dubai","Singapore"):5840,("Dubai","Bangkok"):4900,
    ("Dubai","Hong Kong"):6300,("Dubai","Frankfurt"):5000,("Dubai","New York"):11020,
    ("Dubai","Paris"):5250,("Dubai","Istanbul"):2600,("Dubai","Doha"):350,
    ("Dubai","Ahmedabad"):1950,("Singapore","Hong Kong"):2580,("Singapore","Bangkok"):1450,
    ("Singapore","New York"):15300,("Singapore","Frankfurt"):10360,
    ("Singapore","Paris"):10730,("Singapore","Istanbul"):8000,
    ("Bangkok","Hong Kong"):1730,("Bangkok","New York"):13600,
    ("Bangkok","Frankfurt"):9050,("Bangkok","Istanbul"):7410,
    ("Frankfurt","New York"):6200,("Frankfurt","Istanbul"):2250,
    ("Frankfurt","Hong Kong"):9200,("Frankfurt","Doha"):4770,
    ("Paris","New York"):5840,("Paris","Istanbul"):2240,
    ("Paris","Hong Kong"):9600,("Paris","Doha"):5250,
    ("Istanbul","New York"):8500,("Istanbul","Hong Kong"):8170,
    ("Istanbul","Doha"):3350,("Hong Kong","New York"):12970,
    ("Doha","New York"):11540,("Ahmedabad","Doha"):2000,
    ("Ahmedabad","Dubai"):1950,("Ahmedabad","Bangkok"):3700,
    ("Chennai","Dubai"):2900,("Chennai","Singapore"):3280,
    ("Chennai","Bangkok"):2200,("Bangalore","Dubai"):2900,
    ("Bangalore","Singapore"):3300,("Bangalore","Bangkok"):2150,
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    """Return great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return round(R * 2 * math.asin(math.sqrt(a)))


def lookup_distance(src: str, dst: str) -> Optional[int]:
    """Return known distance (km) for a city pair, or None if not in lookup."""
    return DIST_LOOKUP.get((src, dst)) or DIST_LOOKUP.get((dst, src))


def calculate_distance(src: str, dst: str) -> Tuple[int, str]:
    """
    Return (distance_km, method) where method is 'lookup' or 'haversine'.
    Always returns a value — falls back to haversine via coordinates.
    """
    known = lookup_distance(src, dst)
    if known:
        return known, "lookup"

    if src in CITY_COORDS and dst in CITY_COORDS:
        lat1, lon1 = CITY_COORDS[src]
        lat2, lon2 = CITY_COORDS[dst]
        return haversine(lat1, lon1, lat2, lon2), "haversine"

    # Last resort: index-based estimate
    _cities = sorted(CITY_COORDS.keys())
    fallback = abs(_cities.index(src) - _cities.index(dst)) * 800 + 500
    return fallback, "estimate"


def flight_duration_estimate(distance_km: int, stops: str = "non-stop") -> str:
    """
    Estimate total travel time including layovers.
    Cruise speed ~850 km/h.  Layover time: 1 stop = +2h, 2 stops = +5h.
    """
    layover_hrs = {"non-stop": 0, "1 stop": 2, "2 stops": 5}.get(stops, 0)
    hours_raw   = distance_km / 850 + layover_hrs
    hours = int(hours_raw)
    mins  = round((hours_raw - hours) * 60 / 15) * 15
    if mins == 60:
        hours += 1; mins = 0
    base = f"{hours}h {mins:02d}m" if mins else f"{hours}h"
    if layover_hrs:
        return f"{base} (incl. {layover_hrs}h layover)"
    return base


CITIES = sorted(CITY_COORDS.keys())
