"""Tests for the probit fatality module (Phase 2)."""
import pytest
import numpy as np

from toxic_dispersion.probit import (
    compute_toxic_dose,
    probit_score,
    fatality_probability,
    compute_fatality,
)


def test_compute_toxic_dose():
    """Test toxic dosage integration."""
    # Constant concentration C = 0.1 mg/m³ for 60 seconds
    concentrations = np.full(60, 0.1)
    exposure_time = 60.0

    dosage = compute_toxic_dose(
        concentration_time_series=concentrations,
        exposure_time=exposure_time,
        n_order=1.0,
    )

    # D = C * t = 0.1 * 60 = 6.0
    assert np.isclose(dosage, 6.0, rtol=1e-10)


def test_compute_toxic_dose_power():
    """Test dosage with concentration power n != 1."""
    # C = 0.1 mg/m³ for 60s, with n=2: D = ∫ C^2 dt = 0.01 * 60 = 0.6
    concentrations = np.full(60, 0.1)
    dosage = compute_toxic_dose(
        concentration_time_series=concentrations,
        exposure_time=60.0,
        n_order=2.0,
    )
    assert np.isclose(dosage, 0.6, rtol=1e-10)


def test_probit_score():
    """Test probit score calculation."""
    # Test with known dosage
    y = probit_score(dosage=1.0, k1=-1.04, k2=3.02)
    assert isinstance(y, float)
    assert np.isfinite(y)


def test_fatality_probability():
    """Test fatality probability from probit score."""
    # Y = 0 => P = Phi(-5) ≈ very small
    pf = fatality_probability(y=0.0)
    assert 0.0 <= pf <= 1.0

    # Y = 5 => P = Phi(0) = 0.5
    pf = fatality_probability(y=5.0)
    assert np.isclose(pf, 0.5, rtol=1e-10)


def test_compute_fatality():
    """Test complete fatality computation."""
    result = compute_fatality(
        dosage=10.0,
        chemical="chlorine",
    )

    required_keys = ["dosage", "probit_score", "fatality_probability", "k1", "k2"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    assert result["fatality_probability"] >= 0.0
    assert result["fatality_probability"] <= 1.0
    assert result["probit_score"] > float("-inf")


def test_edge_cases():
    """Test edge cases for probit functions."""
    # Zero or negative dosage should return inf or -inf
    y = probit_score(dosage=0.0, k1=-1.04, k2=3.02)
    assert y == float("-inf") or np.isfinite(y)

    # Negative dosage
    y = probit_score(dosage=-1.0, k1=-1.04, k2=3.02)
    # Should handle gracefully

    # Very large dosage
    pf = fatality_probability(y=100.0)
    assert pf == 1.0 or (pf > 0.0 and pf <= 1.0)