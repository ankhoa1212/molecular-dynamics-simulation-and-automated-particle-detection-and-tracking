"""Tests for temp_graph.py's temperature with the pull's inward motion removed."""

import numpy as np
import pytest

from temp_graph import _compute_frame_temperature

CENTER = 100.0  # the pull center temp_graph.py measures from


def _disk(rng, n, radius):
    """n positions spread uniformly over a disk around the pull center."""
    r = radius * np.sqrt(rng.uniform(size=n))
    theta = rng.uniform(0, 2 * np.pi, size=n)
    return CENTER + r * np.cos(theta), CENTER + r * np.sin(theta)


def _thermal(rng, n, temperature):
    """2D velocities at the given temperature (unit mass)."""
    return rng.normal(0, np.sqrt(temperature), size=(2, n))


def test_thermal_motion_alone_reads_its_temperature():
    rng = np.random.default_rng(0)
    n = 5000
    x, y = _disk(rng, n, 50)
    vx, vy = _thermal(rng, n, 1.0)

    total, thermal = _compute_frame_temperature(x, y, vx, vy, n)

    assert total == pytest.approx(1.0, rel=0.05)
    assert thermal == pytest.approx(1.0, rel=0.05)


def test_linear_inward_flow_is_removed():
    # The pull is a linear spring, so its drift grows with distance: v = -k * r
    rng = np.random.default_rng(1)
    n = 5000
    x, y = _disk(rng, n, 50)
    vx, vy = _thermal(rng, n, 0.5)
    k = 0.2
    vx = vx - k * (x - CENTER)
    vy = vy - k * (y - CENTER)

    total, thermal = _compute_frame_temperature(x, y, vx, vy, n)

    assert total > 20  # the flow dominates the raw kinetic energy
    assert thermal == pytest.approx(0.5, rel=0.1)


def test_atom_at_the_center_is_handled():
    x = np.array([CENTER, 103.0, 97.0])
    y = np.array([CENTER, 100.0, 100.0])
    vx = np.array([1.0, 0.5, -0.5])
    vy = np.array([0.0, 0.2, 0.3])

    total, thermal = _compute_frame_temperature(x, y, vx, vy, 3)

    assert np.isfinite(total) and np.isfinite(thermal)


def test_single_atom_keeps_at_least_one_degree_of_freedom():
    total, thermal = _compute_frame_temperature(
        np.array([110.0]), np.array([100.0]), np.array([0.0]), np.array([2.0]), 1
    )

    # Its radial speed is its own ring's mean, so only the tangential 2.0 is left
    assert total == pytest.approx(2.0)
    assert thermal == pytest.approx(4.0)
