import math

import numpy as np
import pytest

from moga.common.geodesy import GeoOrigin, LocalCartesian, haversine_m
from moga.common.geometry import (PathBuilder, Polyline, Pose2D, box_corners, box_distance,
                                  boxes_overlap, point_in_polygon, wrap_angle)


def test_wrap_angle():
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    a = wrap_angle(np.array([0.0, 2 * math.pi + 0.1, -2 * math.pi - 0.1]))
    np.testing.assert_allclose(a, [0.0, 0.1, -0.1], atol=1e-12)


def test_pose_roundtrip():
    p = Pose2D(10.0, -5.0, 0.7)
    pts = np.random.default_rng(0).normal(size=(20, 2)) * 10
    np.testing.assert_allclose(p.to_local(p.to_world(pts)), pts, atol=1e-9)


def test_polyline_projection_straight():
    pl = Polyline([[0, 0], [100, 0]]).resample(0.5)
    s, d = pl.project(np.array([30.0, 2.0]))
    assert s == pytest.approx(30.0) and d == pytest.approx(2.0)
    s, d = pl.project(np.array([[10.0, -1.0], [120.0, 0.5], [-5.0, 0.0]]))
    np.testing.assert_allclose(s, [10, 120, -5], atol=1e-6)
    np.testing.assert_allclose(d, [-1, 0.5, 0], atol=1e-6)


def test_polyline_arc_curvature_and_frenet_roundtrip():
    pl = PathBuilder().straight(20).arc(50, 90).straight(20).build()
    assert pl.length == pytest.approx(40 + 50 * math.pi / 2, rel=1e-3)
    _, _, yaw, kap = pl.interpolate(20 + 50 * math.pi / 4)
    assert kap == pytest.approx(1 / 50, rel=0.05)
    assert yaw == pytest.approx(math.pi / 4, abs=0.02)
    s = np.linspace(5, pl.length - 5, 50)
    d = np.linspace(-3, 3, 50)
    x, y, _ = pl.to_cartesian(s, d)
    s2, d2 = pl.project(np.column_stack([x, y]))
    np.testing.assert_allclose(s2, s, atol=0.05)
    np.testing.assert_allclose(d2, d, atol=0.02)


def test_clothoid_curve_heading():
    pl = PathBuilder().curve(100, 90, 30).build()
    _, _, yaw, _ = pl.interpolate(pl.length)
    assert yaw == pytest.approx(math.pi / 2, abs=1e-3)


def test_boxes():
    a = box_corners(0, 0, 0, 4, 2)
    b = box_corners(3.5, 0, 0.3, 4, 2)
    c = box_corners(10, 0, 0, 4, 2)
    assert boxes_overlap(a, b)
    assert not boxes_overlap(a, c)
    assert box_distance(a, c) == pytest.approx(6.0)
    assert point_in_polygon(np.array([[0, 0], [5, 5]]), a).tolist() == [True, False]


def test_geodesy_roundtrip():
    lc = LocalCartesian(GeoOrigin(39.92, 32.85, 900.0))
    lat, lon, alt = lc.to_geodetic(1234.5, -987.6, 900.0 - 900.0 + 3.0)
    e, n, u = lc.to_enu(lat, lon, alt)
    assert e == pytest.approx(1234.5, abs=1e-4)
    assert n == pytest.approx(-987.6, abs=1e-4)
    assert u == pytest.approx(3.0, abs=1e-4)
    # metric consistency with great-circle distance
    d = haversine_m(39.92, 32.85, lat, lon)
    assert d == pytest.approx(math.hypot(1234.5, 987.6), rel=3e-3)
