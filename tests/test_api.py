import math

import pytest
from fastapi.testclient import TestClient

from app.geojson import unwrap_longitudes
from app.main import app

client = TestClient(app)

R_KM = 6778.0  # ~400 km circular orbit
V_KMS = math.sqrt(398600.4418 / R_KM)
PERIOD_S = 2 * math.pi * R_KM / V_KMS


def circular_orbit(n: int, inclination_deg: float = 51.6, start_mjd: float = 60000.0):
    inc = math.radians(inclination_deg)
    points = []
    for i in range(n):
        t = PERIOD_S * i / (n - 1)
        u = 2 * math.pi * t / PERIOD_S
        points.append(
            {
                "MjdTime": start_mjd + t / 86400.0,
                "x": R_KM * math.cos(u),
                "y": R_KM * math.sin(u) * math.cos(inc),
                "z": R_KM * math.sin(u) * math.sin(inc),
                "Vx": -V_KMS * math.sin(u),
                "Vy": V_KMS * math.cos(u) * math.cos(inc),
                "Vz": V_KMS * math.cos(u) * math.sin(inc),
            }
        )
    return points


def all_positions(feature):
    return feature["geometry"]["coordinates"]


def test_returns_geojson_feature():
    res = client.post("/orbit/geojson", json=circular_orbit(100))
    assert res.status_code == 200
    body = res.json()
    assert body["type"] == "Feature"
    assert body["geometry"]["type"] == "LineString"
    assert body["properties"]["inputPointCount"] == 100
    # One period (~5554 s) sampled every 5 s, plus the final point
    assert body["properties"]["pointCount"] == math.floor(PERIOD_S / 5) + 2
    assert body["properties"]["stepSeconds"] == 5
    assert body["properties"]["frame"] == "J2000"

    positions = all_positions(body)
    # One coordinate per generated point (time of point i = start + i * step)
    assert len(positions) == body["properties"]["pointCount"]
    assert -180 <= positions[0][0] <= 180
    # Geodetic latitude peaks slightly above the inclination (~51.77 for 51.6)
    assert 51.5 < max(abs(lat) for _, lat, _ in positions) < 51.9
    # Geodetic altitude of a 400 km circular orbit stays within the ellipsoid flattening
    assert all(370_000 < alt < 430_000 for _, _, alt in positions)


def test_full_orbit_is_one_continuous_line():
    body = client.post("/orbit/geojson", json=circular_orbit(200)).json()
    lons = [p[0] for p in body["geometry"]["coordinates"]]
    # No jumps between consecutive points, even across the antimeridian
    assert all(abs(b - a) < 1 for a, b in zip(lons, lons[1:]))
    # A full prograde orbit sweeps ~360 deg minus Earth's rotation (~23 deg),
    # so the line necessarily extends past +/-180
    assert 330 < lons[-1] - lons[0] < 345


def test_points_are_sorted_by_time():
    points = circular_orbit(20)
    body = client.post("/orbit/geojson", json=list(reversed(points))).json()
    assert body["properties"]["startMjd"] == points[0]["MjdTime"]
    assert body["properties"]["endMjd"] == points[-1]["MjdTime"]


def test_meters_unit_matches_km():
    km = circular_orbit(10)
    m = [{k: (v if k == "MjdTime" else v * 1000) for k, v in p.items()} for p in km]
    a = client.post("/orbit/geojson", json=km).json()
    b = client.post("/orbit/geojson?unit=m", json=m).json()
    for pa, pb in zip(all_positions(a), all_positions(b)):
        assert math.isclose(pa[0], pb[0], abs_tol=1e-9)
        assert math.isclose(pa[1], pb[1], abs_tol=1e-9)
        assert math.isclose(pa[2], pb[2], abs_tol=1e-3)


def test_empty_list_rejected():
    assert client.post("/orbit/geojson", json=[]).status_code == 422


def test_missing_field_rejected():
    point = circular_orbit(2)[0]
    del point["z"]
    assert client.post("/orbit/geojson", json=[point]).status_code == 422


def test_unwrap_longitudes_eastward_crossing():
    assert unwrap_longitudes([[170.0, 0.0, 0.0], [-170.0, 10.0, 100.0], [-160.0, 20.0, 0.0]]) == [
        [170.0, 0.0, 0.0],
        [190.0, 10.0, 100.0],
        [200.0, 20.0, 0.0],
    ]


def test_unwrap_longitudes_westward_crossing():
    assert unwrap_longitudes([[-175.0, 0.0, 0.0], [165.0, -20.0, 0.0]]) == [
        [-175.0, 0.0, 0.0],
        [-195.0, -20.0, 0.0],
    ]


def test_unwrap_longitudes_no_crossing_unchanged():
    positions = [[10.0, 0.0, 0.0], [20.0, 5.0, 0.0], [15.0, 10.0, 0.0]]
    assert unwrap_longitudes(positions) == positions


def test_custom_step_seconds():
    body = client.post("/orbit/geojson?step_seconds=60", json=circular_orbit(10)).json()
    assert body["properties"]["pointCount"] == math.floor(PERIOD_S / 60) + 2


def test_invalid_step_rejected():
    assert client.post("/orbit/geojson?step_seconds=0", json=circular_orbit(2)).status_code == 422


def test_too_many_samples_rejected():
    points = circular_orbit(2)
    points[1]["MjdTime"] = points[0]["MjdTime"] + 30  # 30 days at 5 s
    res = client.post("/orbit/geojson", json=points)
    assert res.status_code == 422
    assert "step_seconds" in res.json()["detail"]


def test_duplicate_times_ignored():
    points = circular_orbit(10)
    body = client.post("/orbit/geojson", json=points + [points[3]]).json()
    assert body["properties"]["inputPointCount"] == 10
