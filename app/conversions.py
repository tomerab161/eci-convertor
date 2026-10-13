"""ECI -> ECEF -> geodetic (WGS84) conversions.

Pure-math implementation (no astro libraries):
- Time: MJD is treated as UTC and used as an approximation of UT1/TT
  (error < 1 s, i.e. < ~0.5 km along the ground track).
- J2000 -> mean-of-date: IAU-1976 precession.
- Mean-of-date / TEME -> ECEF: rotation by GMST (IAU-1982).
- Nutation and polar motion are ignored (< ~20 arcsec, i.e. < ~0.6 km).
"""

import math
from enum import Enum

MJD_TO_JD = 2400000.5
JD_J2000 = 2451545.0
DAYS_PER_CENTURY = 36525.0
ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)

# WGS84 ellipsoid
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


class EciFrame(str, Enum):
    J2000 = "J2000"
    TEME = "TEME"


class DistanceUnit(str, Enum):
    KM = "km"
    M = "m"


UNIT_TO_METERS = {DistanceUnit.KM: 1000.0, DistanceUnit.M: 1.0}

Vector = tuple[float, float, float]


def _rot_z(angle: float, v: Vector) -> Vector:
    """Frame rotation about Z (Vallado's R3)."""
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = v
    return (c * x + s * y, -s * x + c * y, z)


def _rot_y(angle: float, v: Vector) -> Vector:
    """Frame rotation about Y (Vallado's R2)."""
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = v
    return (c * x - s * z, y, s * x + c * z)


def _julian_centuries(mjd: float) -> float:
    return (mjd + MJD_TO_JD - JD_J2000) / DAYS_PER_CENTURY


def gmst_rad(mjd: float) -> float:
    """Greenwich Mean Sidereal Time (IAU-1982) in radians, in [0, 2*pi)."""
    t = _julian_centuries(mjd)
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t**2
        - 6.2e-6 * t**3
    )
    return math.radians((seconds % 86400.0) / 240.0)


def precess_j2000_to_mean_of_date(r: Vector, mjd: float) -> Vector:
    """Apply IAU-1976 precession: J2000 mean equator/equinox -> mean of date."""
    t = _julian_centuries(mjd)
    zeta = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * ARCSEC_TO_RAD
    theta = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * ARCSEC_TO_RAD
    z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * ARCSEC_TO_RAD
    return _rot_z(-z, _rot_y(theta, _rot_z(-zeta, r)))


def eci_to_ecef(r: Vector, mjd: float, frame: EciFrame = EciFrame.J2000) -> Vector:
    if frame is EciFrame.J2000:
        r = precess_j2000_to_mean_of_date(r, mjd)
    return _rot_z(gmst_rad(mjd), r)


def ecef_to_geodetic(r_m: Vector) -> tuple[float, float, float]:
    """ECEF (meters) -> (longitude deg, latitude deg, altitude m) on WGS84."""
    x, y, z = r_m
    lon = math.atan2(y, x)
    p = math.hypot(x, y)

    if p < 1e-9:  # On the polar axis
        b = WGS84_A_M * (1.0 - WGS84_F)
        return 0.0, math.copysign(90.0, z), abs(z) - b

    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        new_lat = math.atan2(z + WGS84_E2 * n * sin_lat, p)
        if abs(new_lat - lat) < 1e-12:
            lat = new_lat
            break
        lat = new_lat

    sin_lat = math.sin(lat)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    # Use the better-conditioned formula depending on latitude
    if abs(lat) < math.radians(45.0):
        alt = p / math.cos(lat) - n
    else:
        alt = z / sin_lat - n * (1.0 - WGS84_E2)

    return math.degrees(lon), math.degrees(lat), alt


def eci_to_geodetic(
    r: Vector,
    mjd: float,
    frame: EciFrame = EciFrame.J2000,
    unit: DistanceUnit = DistanceUnit.KM,
) -> tuple[float, float, float]:
    """ECI position -> (longitude deg, latitude deg, altitude m)."""
    scale = UNIT_TO_METERS[unit]
    x, y, z = eci_to_ecef(r, mjd, frame)
    return ecef_to_geodetic((x * scale, y * scale, z * scale))
