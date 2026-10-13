import math

import pytest

from app.interpolation import SECONDS_PER_DAY, resample, sample_count
from app.models import OrbitalPoint

MU = 398600.4418
R_KM = 6778.0
W = math.sqrt(MU / R_KM**3)  # angular rate, rad/s
START_MJD = 60000.0


def circular_point(t: float) -> OrbitalPoint:
    """Equatorial circular orbit, t seconds after START_MJD."""
    u = W * t
    return OrbitalPoint(
        MjdTime=START_MJD + t / SECONDS_PER_DAY,
        x=R_KM * math.cos(u),
        y=R_KM * math.sin(u),
        z=0.0,
        Vx=-R_KM * W * math.sin(u),
        Vy=R_KM * W * math.cos(u),
        Vz=0.0,
    )


def test_samples_every_step_including_both_ends():
    points = [circular_point(t) for t in (0, 60, 120)]
    samples = resample(points, 5.0)
    assert len(samples) == sample_count(points, 5.0) == 25
    offsets = [(mjd - START_MJD) * SECONDS_PER_DAY for mjd, _ in samples]
    assert offsets == pytest.approx([5.0 * k for k in range(25)], abs=1e-4)


def test_last_point_appended_when_span_not_multiple_of_step():
    points = [circular_point(t) for t in (0, 12)]
    samples = resample(points, 5.0)
    offsets = [(mjd - START_MJD) * SECONDS_PER_DAY for mjd, _ in samples]
    assert offsets == pytest.approx([0, 5, 10, 12], abs=1e-4)
    assert samples[-1][1] == pytest.approx((points[-1].x, points[-1].y, points[-1].z))


def test_passes_through_input_points():
    points = [circular_point(t) for t in (0, 60, 120)]
    samples = resample(points, 5.0)
    for p, (_, pos) in zip(points, samples[::12]):
        # MJD doubles carry ~µs time resolution -> mm-level position jitter
        assert pos == pytest.approx((p.x, p.y, p.z), abs=1e-5)


def test_hermite_follows_true_orbit():
    # Input points 60 s apart; generated points must match the analytic orbit
    points = [circular_point(t) for t in range(0, 600, 60)]
    for mjd, pos in resample(points, 5.0):
        t = (mjd - START_MJD) * SECONDS_PER_DAY
        truth = circular_point(t)
        err = math.dist(pos, (truth.x, truth.y, truth.z))
        assert err < 0.01  # km


def test_single_point():
    p = circular_point(0)
    assert resample([p], 5.0) == [(p.MjdTime, (p.x, p.y, p.z))]
