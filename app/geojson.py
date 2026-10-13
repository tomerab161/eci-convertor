"""Build a GeoJSON orbit (ground track) from orbital points."""

from datetime import datetime, timedelta, timezone
from typing import Any

from app.conversions import DistanceUnit, EciFrame, eci_to_geodetic
from app.interpolation import resample
from app.models import OrbitalPoint

Position = list[float]  # [lon, lat, alt_m]

MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)


def _mjd_to_iso(mjd: float) -> str:
    return (MJD_EPOCH + timedelta(days=mjd)).isoformat().replace("+00:00", "Z")


def unwrap_longitudes(positions: list[Position]) -> list[Position]:
    """Make longitude continuous so the track is one unbroken line.

    Instead of jumping from +180 to -180 when crossing the antimeridian, the
    longitude keeps going (181, 182, ...), and likewise below -180. The first
    point stays within [-180, 180]. Web maps (Leaflet, Mapbox GL, OpenLayers)
    draw such a line continuously across the antimeridian.
    """
    unwrapped: list[Position] = []
    for lon, lat, alt in positions:
        if unwrapped:
            prev_lon = unwrapped[-1][0]
            # Shortest signed step, in [-180, 180)
            d_lon = (lon - prev_lon + 180.0) % 360.0 - 180.0
            lon = prev_lon + d_lon
        unwrapped.append([lon, lat, alt])
    return unwrapped


def prepare_points(points: list[OrbitalPoint]) -> list[OrbitalPoint]:
    """Sort by time and drop points with duplicate times (first one wins)."""
    ordered: list[OrbitalPoint] = []
    for p in sorted(points, key=lambda p: p.MjdTime):
        if not ordered or p.MjdTime != ordered[-1].MjdTime:
            ordered.append(p)
    return ordered


def orbit_to_geojson(
    points: list[OrbitalPoint],
    frame: EciFrame = EciFrame.J2000,
    unit: DistanceUnit = DistanceUnit.KM,
    step_seconds: float = 5.0,
) -> dict[str, Any]:
    """points must already be passed through prepare_points()."""
    samples = resample(points, step_seconds)
    positions = [list(eci_to_geodetic(r, mjd, frame, unit)) for mjd, r in samples]

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": unwrap_longitudes(positions),
        },
        "properties": {
            "frame": frame.value,
            "inputUnit": unit.value,
            "stepSeconds": step_seconds,
            "inputPointCount": len(points),
            "pointCount": len(samples),
            "startMjd": points[0].MjdTime,
            "endMjd": points[-1].MjdTime,
            "startTime": _mjd_to_iso(points[0].MjdTime),
            "endTime": _mjd_to_iso(points[-1].MjdTime),
        },
    }
