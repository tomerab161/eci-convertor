from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query

from app.conversions import DistanceUnit, EciFrame
from app.geojson import orbit_to_geojson, prepare_points
from app.interpolation import sample_count
from app.models import OrbitalPoint

# ~11.5 days at a 5 s step
MAX_SAMPLES = 200_000

app = FastAPI(title="ECI Convertor", version="1.0.0")


@app.post("/orbit/geojson", summary="Convert ECI orbital points to a GeoJSON orbit")
def orbital_points_to_geojson(
    points: list[OrbitalPoint] = Body(min_length=1),
    frame: EciFrame = Query(EciFrame.J2000, description="ECI frame of the input"),
    unit: DistanceUnit = Query(
        DistanceUnit.KM, description="Unit of x, y, z (velocities in unit per second)"
    ),
    step_seconds: float = Query(
        5.0, gt=0, description="Time between generated orbit points, in seconds"
    ),
) -> dict[str, Any]:
    """Interpolate the orbital points every `step_seconds` and return the ground
    track as a GeoJSON Feature with a single LineString. Coordinates are
    [longitude, latitude, altitude_m]; longitude is unwrapped (may go beyond
    +/-180) so the line stays continuous across the antimeridian."""
    ordered = prepare_points(points)
    count = sample_count(ordered, step_seconds)
    if count > MAX_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=f"Orbit would have {count} points (max {MAX_SAMPLES}); "
            "increase step_seconds or shorten the time span",
        )
    return orbit_to_geojson(ordered, frame, unit, step_seconds)
