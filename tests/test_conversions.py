import math

import pytest

from app.conversions import (
    WGS84_A_M,
    WGS84_F,
    EciFrame,
    ecef_to_geodetic,
    eci_to_ecef,
    gmst_rad,
)

# 1992-08-20 12:14:00 UT1 (Vallado, Example 3-5): GMST = 152.578787810 deg
VALLADO_MJD = 48854.0 + (12 * 3600 + 14 * 60) / 86400.0


def test_gmst_at_j2000_epoch():
    assert math.degrees(gmst_rad(51544.5)) == pytest.approx(280.46061837, abs=1e-6)


def test_gmst_vallado_example():
    assert math.degrees(gmst_rad(VALLADO_MJD)) == pytest.approx(152.578787810, abs=1e-6)


def test_teme_rotation_by_gmst():
    # A point in the direction of GMST longitude in ECI is over the prime meridian
    g = gmst_rad(VALLADO_MJD)
    x, y, z = eci_to_ecef((math.cos(g), math.sin(g), 0.0), VALLADO_MJD, EciFrame.TEME)
    assert (x, y, z) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)


def test_j2000_precession_is_identity_at_epoch():
    r = (7000.0, 1234.0, -500.0)
    assert eci_to_ecef(r, 51544.5, EciFrame.J2000) == pytest.approx(
        eci_to_ecef(r, 51544.5, EciFrame.TEME), abs=1e-6
    )


def test_j2000_precession_magnitude_after_26_years():
    # General precession in RA at the equator is ~46.1"/yr -> ~0.333 deg over
    # 26 years; a fixed J2000 direction gains RA in the frame of date
    mjd = 51544.5 + 26 * 365.25
    r = (7000.0, 0.0, 0.0)
    teme = eci_to_ecef(r, mjd, EciFrame.TEME)
    j2000 = eci_to_ecef(r, mjd, EciFrame.J2000)
    angle = math.degrees(math.atan2(j2000[1], j2000[0]) - math.atan2(teme[1], teme[0]))
    assert angle == pytest.approx(0.333, abs=0.002)


@pytest.mark.parametrize(
    "lon, lat, alt",
    [(0, 0, 0), (45, 30, 500_000), (-120, -60, 35_786_000), (179.9, 89.5, 400_000)],
)
def test_geodetic_roundtrip(lon, lat, alt):
    e2 = WGS84_F * (2 - WGS84_F)
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A_M / math.sqrt(1 - e2 * math.sin(la) ** 2)
    ecef = (
        (n + alt) * math.cos(la) * math.cos(lo),
        (n + alt) * math.cos(la) * math.sin(lo),
        (n * (1 - e2) + alt) * math.sin(la),
    )
    assert ecef_to_geodetic(ecef) == pytest.approx((lon, lat, alt), abs=1e-6)


def test_geodetic_on_pole():
    b = WGS84_A_M * (1 - WGS84_F)
    assert ecef_to_geodetic((0.0, 0.0, b + 1000)) == pytest.approx((0, 90, 1000))
