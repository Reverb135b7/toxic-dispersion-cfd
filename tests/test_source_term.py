"""Tests for the source-term and flash evaporation module (Phase 1)."""
import pytest
import numpy as np

from toxic_dispersion.source_term import (
    flash_vapor_fraction,
    pool_spreading_rate,
    evaporation_rate,
    compute_flash_and_evaporation,
)


def test_flash_vapor_fraction():
    """Test flash vapor fraction calculation."""
    # Test with T_storage > T_bp: should give positive fraction
    x = flash_vapor_fraction(
        T_storage=15.0,       # °C, above ammonia bp (-33°C)
        T_bp=-33.34,
        delta_h_vap=23.35e3,
        cp_l=4.7,
    )
    assert 0.0 < x <= 1.0

    # Test with T_storage = T_bp: should give zero
    x = flash_vapor_fraction(
        T_storage=-33.34,
        T_bp=-33.34,
        delta_h_vap=23.35e3,
        cp_l=4.7,
    )
    assert x == 0.0

    # Test with T_storage < T_bp: should give zero (clamped)
    x = flash_vapor_fraction(
        T_storage=-50.0,
        T_bp=-33.34,
        delta_h_vap=23.35e3,
        cp_l=4.7,
    )
    assert x == 0.0


def test_pool_spreading_rate():
    """Test pool radius growth rate calculation."""
    # Water-like parameters
    rate = pool_spreading_rate(
        r_p=1.0,
        g=9.81,
        rho_l=1000.0,
        rho_a=1.2,
        y_p=1.0,
    )
    # Expected: sqrt(2 * 9.81 * (1000-1.2)/1000 * 1) ≈ sqrt(19.6) ≈ 4.43
    assert np.isclose(rate, np.sqrt(2 * 9.81 * (1000.0 - 1.2) / 1000.0), rtol=1e-5)


def test_evaporation_rate():
    """Test evaporation mass loss rate calculation."""
    # Use psat > Tambient to get positive evaporation rate
    # With psat=105000 Pa (~1.04 bar) and Tambient=101325 Pa
    mdot = evaporation_rate(
        m=17.031,      # ammonia MW
        kg=0.001,      # mass transfer coeff
        ap=np.pi * 1.0**2,  # area for r=1m
        psat=105000.0,  # saturation pressure > ambient
        Tambient=101325.0,
        tp=15.0,
        r=1.0,
    )
    # Should produce a positive rate
    assert mdot > 0


def test_compute_flash_and_evaporation():
    """Test full source-term computation."""
    result = compute_flash_and_evaporation(
        chemical="ammonia",
        T_storage=15.0,
        P_storage=50.0,  # bar
        mass_liquid=1000.0,
    )

    # Check required keys exist
    required_keys = ["x_flash", "t_bp", "mw", "rho_l", "p_sat", "dotm_evap", "r_initial"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # Flash fraction should be in [0, 1]
    assert 0.0 <= result["x_flash"] <= 1.0

    # Boiling point should be ammonia's bp
    assert result["t_bp"] == -33.34

    # Molecular weight should be ammonia's mw
    assert result["mw"] == 17.031

    # Saturation pressure should be positive
    assert result["p_sat"] > 0

    # Evaporation rate should be positive
    assert result["dotm_evap"] > 0

    # Initial radius should be positive
    assert result["r_initial"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])