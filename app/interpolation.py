"""Resample orbital points at a fixed time step.

Between consecutive points the ECI position is interpolated with a cubic
Hermite spline, which matches both position and velocity at each end. This is
done in the inertial frame, where the motion is smooth, before converting to
Earth-fixed coordinates.

Velocities must use the position unit per second (e.g. km and km/s).
"""

import math
from bisect import bisect_right

from app.models import OrbitalPoint

SECONDS_PER_DAY = 86400.0

Vector = tuple[float, float, float]


def _hermite(p0: Vector, v0: Vector, p1: Vector, v1: Vector, dt: float, s: float) -> Vector:
    """Position at fraction s in [0, 1] of an interval of dt seconds."""
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return tuple(
        h00 * a + h10 * dt * va + h01 * b + h11 * dt * vb
        for a, va, b, vb in zip(p0, v0, p1, v1)
    )


def sample_count(points: list[OrbitalPoint], step_seconds: float) -> int:
    """Number of samples resample() will produce for already-sorted points."""
    duration = (points[-1].MjdTime - points[0].MjdTime) * SECONDS_PER_DAY
    n = math.floor(duration / step_seconds + 1e-9) + 1
    return n + (1 if (n - 1) * step_seconds < duration - 1e-6 else 0)


def resample(
    points: list[OrbitalPoint], step_seconds: float
) -> list[tuple[float, Vector]]:
    """Return (mjd, eci_position) every step_seconds from the first point to
    the last (the last point is always included). Points must be sorted by time
    with no duplicate times."""
    t0 = points[0].MjdTime
    if len(points) == 1:
        return [(t0, (points[0].x, points[0].y, points[0].z))]

    offsets = [(p.MjdTime - t0) * SECONDS_PER_DAY for p in points]
    duration = offsets[-1]

    count = sample_count(points, step_seconds)
    sample_offsets = [min(k * step_seconds, duration) for k in range(count)]

    samples = []
    for t in sample_offsets:
        i = min(bisect_right(offsets, t) - 1, len(points) - 2)
        a, b = points[i], points[i + 1]
        dt = offsets[i + 1] - offsets[i]
        if dt <= 0:
            pos = (a.x, a.y, a.z)
        else:
            pos = _hermite(
                (a.x, a.y, a.z), (a.Vx, a.Vy, a.Vz),
                (b.x, b.y, b.z), (b.Vx, b.Vy, b.Vz),
                dt, (t - offsets[i]) / dt,
            )
        samples.append((t0 + t / SECONDS_PER_DAY, pos))
    return samples
