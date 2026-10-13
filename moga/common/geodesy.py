"""WGS-84 geodesy: geodetic <-> ECEF <-> local ENU conversions.

The HD map stores a geodetic origin; the GNSS receiver reports latitude /
longitude / altitude and the autonomy stack converts them into the local ENU
map frame with :class:`LocalCartesian`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)


def geodetic_to_ecef(lat_deg, lon_deg, alt):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    x = (n + alt) * cl * np.cos(lon)
    y = (n + alt) * cl * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sl
    return x, y, z


def ecef_to_geodetic(x, y, z):
    """Bowring's method followed by two Newton refinements (sub-mm accuracy)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3,
                     p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3)
    for _ in range(2):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


@dataclass(frozen=True)
class GeoOrigin:
    lat: float
    lon: float
    alt: float = 0.0


class LocalCartesian:
    """Local tangent plane (ENU) anchored at a geodetic origin."""

    def __init__(self, origin: GeoOrigin):
        self.origin = origin
        lat0 = math.radians(origin.lat)
        lon0 = math.radians(origin.lon)
        sl, cl = math.sin(lat0), math.cos(lat0)
        so, co = math.sin(lon0), math.cos(lon0)
        # rows: east, north, up expressed in ECEF
        self.R = np.array([[-so, co, 0.0],
                           [-sl * co, -sl * so, cl],
                           [cl * co, cl * so, sl]])
        self.ecef0 = np.array(geodetic_to_ecef(origin.lat, origin.lon, origin.alt))

    def to_enu(self, lat, lon, alt=0.0):
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        d = np.stack([np.asarray(x) - self.ecef0[0], np.asarray(y) - self.ecef0[1],
                      np.asarray(z) - self.ecef0[2]])
        enu = self.R @ d.reshape(3, -1)
        if np.ndim(lat) == 0:
            return float(enu[0, 0]), float(enu[1, 0]), float(enu[2, 0])
        return enu[0], enu[1], enu[2]

    def to_geodetic(self, e, n, u=0.0):
        enu = np.stack([np.asarray(e, float), np.asarray(n, float),
                        np.broadcast_to(np.asarray(u, float), np.shape(e))]).reshape(3, -1)
        d = self.R.T @ enu
        lat, lon, alt = ecef_to_geodetic(d[0] + self.ecef0[0], d[1] + self.ecef0[1],
                                         d[2] + self.ecef0[2])
        if np.ndim(e) == 0:
            return float(lat[0]), float(lon[0]), float(alt[0])
        return lat, lon, alt


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))
