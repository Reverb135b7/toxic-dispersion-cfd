"""Source-term and flash evaporation module (Phase 1).

Implements the Modified Brighton-Evans model for flash vapor fraction,
pool spreading, and dynamic vapor generation rate calculation.
"""
import numpy as np
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Chemical property database (critical properties, boiling points, heat of
# vaporization, liquid densities at storage conditions).
# ---------------------------------------------------------------------------

import math

CHEMICAL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "ammonia": {
        "mw": 17.031,          # g/mol
        "t_bp": -33.34,        # °C boiling point at 1 atm
        "delta_h_vap": 23.35e3,  # J/mol heat of vaporization
        "cp_l": 4.7,           # J/g·K liquid specific heat
        "rho_l_20": 681.0,     # kg/m³ liquid density at 20°C
        "p_sat_coeff": 61.7,   # Antoine A parameter
        "p_sat_b": 13.28,      # Antoine B parameter
        "p_sat_c": -5.2,       # Antoine C parameter
    },
    "chlorine": {
        "mw": 70.906,
        "t_bp": -34.05,
        "delta_h_vap": 20.28e3,
        "cp_l": 0.48,          # kJ/kg·K ≈ 480 J/kg·K
        "rho_l_20": 1639.0,
        "p_sat_coeff": 12.0,
        "p_sat_b": 2.75,
        "p_sat_c": -3.8,
    },
    "hydrogen_sulfide": {
        "mw": 34.081,
        "t_bp": -60.0,
        "delta_h_vap": 18.60e3,
        "cp_l": 1.36,          # kJ/kg·K ≈ 1360 J/kg·K
        "rho_l_20": 968.0,
        "p_sat_coeff": 5.8,
        "p_sat_b": 1.5,
        "p_sat_c": -2.5,
    },
    "hydrogen_fluoride": {
        "mw": 20.006,
        "t_bp": 19.5,          # °C boiling point at 1 atm
        "delta_h_vap": 7.24e3,   # J/mol heat of vaporization
        "cp_l": 2.66,          # J/g·K liquid specific heat
        "rho_l_20": 974.0,     # kg/m³ liquid density at 20°C
        "p_sat_coeff": 15.5,   # Antoine A parameter
        "p_sat_b": 3910.0,   # Antoine B parameter
        "p_sat_c": -53.0,    # Antoine C parameter
    },
    "phosgene": {
        "mw": 98.99,
        "t_bp": 8.2,           # °C boiling point at 1 atm
        "delta_h_vap": 24.0e3,  # J/mol heat of vaporization
        "cp_l": 1.0,           # J/g·K approximate
        "rho_l_20": 1350.0,    # kg/m³ approximate
        "p_sat_coeff": 4.2,    # Antoine A parameter
        "p_sat_b": 1470.0,   # Antoine B parameter
        "p_sat_c": -32.0,    # Antoine C parameter
    },
    "sulfur_dioxide": {
        "mw": 64.06,
        "t_bp": -10.0,         # °C boiling point at 1 atm
        "delta_h_vap": 24.9e3,  # J/mol heat of vaporization
        "cp_l": 0.64,          # J/g·K approximate
        "rho_l_20": 1430.0,    # kg/m³ approximate
        "p_sat_coeff": 8.68,   # Antoine A parameter
        "p_sat_b": 2370.0,   # Antoine B parameter
        "p_sat_c": -34.0,    # Antoine C parameter
    },
}


def jet_momentum_flash_fraction(
    T_storage: float,
    P_storage: float,
    T_bp: float,
    delta_h_vap: float,
    cp_l: float,
    mw: float,
    g: float = 9.81,
) -> Dict[str, float]:
    """Calculate flash vapor fraction with jet momentum correction.

    For high-pressure releases, the jet velocity affects entrainment
    and early-time dispersion. Uses the Brighton-Evans base with
    a momentum correction factor.

    Parameters
    ----------
    T_storage : float
        Storage temperature in °C.
    P_storage : float
        Storage pressure in Pa.
    T_bp : float
        Normal boiling point in °C.
    delta_h_vap : float
        Heat of vaporization in J/mol.
    cp_l : float
        Liquid specific heat in J/g·K.
    mw : float
        Molecular weight in g/mol.
    g : float, optional
        Gravitational acceleration in m/s² (default 9.81).

    Returns
    -------
    Dict[str, float]
        Dictionary with 'x_flash_base' (Brighton-Evans fraction) and
        'jet_factor' (momentum entrainment enhancement).
    """
    # Base Brighton-Evans flash fraction
    x_flash_base = cp_l * (T_storage - T_bp) / delta_h_vap
    x_flash_base = max(0.0, min(1.0, x_flash_base))

    # Jet momentum factor: higher pressure => more entrainment
    # Simplified: factor = 1 + alpha * (P_storage / P_atm)^beta
    # alpha, beta empirical; typical values ~0.1, ~0.5
    patm = 101325.0  # Pa ambient pressure
    pressure_ratio = P_storage / patm
    jet_factor = 1.0 + 0.15 * (pressure_ratio ** 0.5)

    # Effective flash fraction enhanced by jet entrainment
    x_flash_eff = min(1.0, x_flash_base * jet_factor)

    # Estimate initial jet centerline velocity (order-of-magnitude)
    # Using isenthalpic flash calculation approximation
    r_gas = 8314.462  # J/(kmol·K)
    T_kelvin = T_storage + 273.15
    # Mass flux approximation: G = P0 * sqrt( (2*gamma)/(R*T0*(gamma-1)) * (1 - (Pe/P0)^((gamma-1)/gamma) ) )
    # For simplified estimate, use orifice flow
    discharge_coeff = 0.8  # typical for sharp-edged orifice
    mw_kg_mol = mw / 1000.0  # kg/mol
    # Rough jet velocity estimate
    if P_storage > patm:
        # Two-phase jet velocity scaling
        jet_vel_est = discharge_coeff * math.sqrt(2.0 * g * (T_kelvin / 273.15) * (P_storage / patm - 1.0))
    else:
        jet_vel_est = 0.0

    return {
        "x_flash_base": x_flash_base,
        "jet_factor": jet_factor,
        "x_flash_eff": x_flash_eff,
        "jet_vel_est": jet_vel_est,
    }


def compute_flash_with_jet(
    chemical: str,
    T_storage: float,
    P_storage: float,
    mass_liquid: float,
    ambient_temperature: float = 25.0,
    ambient_pressure: float = 101325.0,
) -> Dict[str, Optional[float]]:
    """Compute flash evaporation with jet momentum enhancement (Phase 1 advanced).

    Parameters
    ----------
    chemical : str
        Chemical key from CHEMICAL_PROPERTIES dict.
    T_storage : float
        Storage temperature in °C.
    P_storage : float
        Storage pressure in Pa.
    mass_liquid : float
        Initial liquid mass in kg.
    ambient_temperature : float
        Ambient air temperature in °C (default 25°C).
    ambient_pressure : float
        Ambient pressure in Pa (default 101325 Pa).

    Returns
    -------
    Dict[str, Optional[float]]
        Dictionary with flash and pool parameters including jet momentum:
        - 'x_flash_base': Base Brighton-Evans fraction
        - 'jet_factor': Momentum entrainment enhancement factor
        - 'x_flash_eff': Effective flash fraction
        - 'jet_vel_est': Estimated jet centerline velocity in m/s
        - 't_bp': Boiling point in °C
        - 'mw': Molecular weight in g/mol
        - 'rho_l': Liquid density at storage condition (estimated) in kg/m³
        - 'p_sat': Saturation pressure at T_storage in Pa
        - 'dotm_evap': Initial evaporation rate in kg/s (estimated)
        - 'r_initial': Initial pool radius in meters
    """
    props = CHEMICAL_PROPERTIES.get(chemical)
    if props is None:
        raise ValueError(f"Unknown chemical: {chemical}. Available: {list(CHEMICAL_PROPERTIES.keys())}")

    T_bp = props["t_bp"]
    delta_h_vap = props["delta_h_vap"]
    cp_l = props["cp_l"]
    mw = props["mw"]
    rho_l_20 = props["rho_l_20"]

    # 1. Jet-enhanced flash calculation
    jet_result = jet_momentum_flash_fraction(
        T_storage, P_storage, T_bp, delta_h_vap, cp_l, mw,
    )

    # 2. Estimate liquid pool temperature (assume subcooled)
    T_pool = min(T_storage, T_bp - 1.0)  # stay below bp

    # 3. Saturation pressure at pool temperature using Antoine
    a, b, c = props["p_sat_coeff"], props["p_sat_b"], props["p_sat_c"]
    p_sat_bar = 10.0 ** (a - b / (c + T_pool))
    p_sat = p_sat_bar * 100000.0  # bar to Pa

    # 3. Liquid density adjustment (rough constant for now)
    rho_l = rho_l_20  # kg/m³

    # 4. Initial pool radius from mass and density
    # Assuming circular pool, depth ~ 0.1 m for initial spread
    pool_depth = 0.1  # m assumed initial depth
    r_initial = math.sqrt(mass_liquid / (rho_l * pool_depth * math.pi))

    # 5. Initial evaporation rate with jet enhancement
    # Kg ~ 0.001 m/s for still air, enhanced by jet factor
    kg_approx = 0.001 * jet_result["jet_factor"]  # m/s approximate
    ap = math.pi * r_initial ** 2  # m²
    # mdot = (M/1000) * Kg * Ap * (Psat - P_amb) / (R * T)
    r_spec = 8314.462 / mw  # J/(kg·K) specific gas constant
    mdot = (mw / 1000.0) * kg_approx * ap * (p_sat - ambient_pressure) / (r_spec * (T_pool + 273.15))

    result = {
        "x_flash_base": jet_result["x_flash_base"],
        "jet_factor": jet_result["jet_factor"],
        "x_flash_eff": jet_result["x_flash_eff"],
        "jet_vel_est": jet_result["jet_vel_est"],
        "t_bp": T_bp,
        "mw": mw,
        "rho_l": rho_l,
        "p_sat": p_sat,
        "dotm_evap": mdot,
        "r_initial": r_initial,
    }

    return result



def antoine_pvp(T: float, a: float, b: float, c: float) -> float:
    """Calculate saturation pressure using Antoine equation.

    log10(P) = A - B / (C + T)  with P in bar, T in °C.

    Returns pressure in bar.
    """
    return 10.0 ** (a - b / (c + T))


def flash_vapor_fraction(
    T_storage: float,
    T_bp: float,
    delta_h_vap: float,
    cp_l: float,
) -> float:
    """Calculate flash vapor fraction using modified Brighton-Evans model.

    x_flash = Cp,l * (T_storage - T_bp) / ΔH_vap

    Parameters
    ----------
    T_storage : float
        Storage temperature in °C.
    T_bp : float
        Normal boiling point in °C.
    delta_h_vap : float
        Heat of vaporization in J/mol.
    cp_l : float
        Liquid specific heat in J/g·K.

    Returns
    -------
    float
        Flash vapor fraction (0 <= x_flash <= 1).
    """
    x_flash = cp_l * (T_storage - T_bp) / delta_h_vap
    # Clamp to physical range
    return max(0.0, min(1.0, x_flash))


def pool_spreading_rate(
    r_p: float,
    g: float,
    rho_l: float,
    rho_a: float,
    y_p: float,
) -> float:
    """Calculate pool radius growth rate.

    dR_p/dt = sqrt(2 * g * (rho_l - rho_a) / rho_l * y_p)

    Parameters
    ----------
    r_p : float
        Current pool radius in meters.
    g : float
        Gravitational acceleration in m/s².
    rho_l : float
        Liquid density in kg/m³.
    rho_a : float
        Ambient air density in kg/m³.
    y_p : float
        Pool mass fraction (0 < y_p <= 1).

    Returns
    -------
    float
        Pool radius growth rate dr_p/dt in m/s.
    """
    return np.sqrt(2.0 * g * (rho_l - rho_a) / rho_l * y_p)


def evaporation_rate(
    m: float,
    kg: float,
    ap: float,
    psat: float,
   Tambient: float,
    tp: float,
    r: float,
) -> float:
    """Calculate evaporative mass loss rate.

    m_evap = M * Kg * Ap * (Psat(Tp) - P_amb) / (R * Tp)

    Parameters
    ----------
    m : float
        Molecular weight in g/mol.
    kg : float
        Mass transfer coefficient in m/s.
    ap : float
        Pool surface area in m².
    psat : float
        Saturation pressure at pool temperature in Pa.
    t_ambient : float
        Ambient pressure in Pa (default ~101325 Pa).
    tp : float
        Pool temperature in °C (used for Antoine).
    r : float
        Pool radius in meters (for area calculation if needed).

    Returns
    -------
    float
        Evaporation rate in kg/s.
    """
    r_gas = 8314.462  # Universal gas constant J/(kmol·K)
    # Convert molecular weight from g/mol to kg/kmol
    mw_kg_kmol = m  # numerically same, but conceptually kg/kmol
    # Note: The formula uses M in g/mol with R in J/(mol·K) if consistent
    # We'll use R = 8314 J/(kmol·K) and m in g/mol → convert
    # Actually let's use R = 8.314 J/(mol·K) and m in g/mol → convert to kg
    # Simpler: use R_specific = R_universal / mw
    r_spec = r_gas / m  # J/(kg·K)
    # Evaporation rate
    mdot = mw_kg_kmol * kg * ap * (psat - Tambient) / (r_spec * (tp + 273.15))
    # Wait, let's re-derive properly:
    # m_evap = M * Kg * Ap * (Psat - P_amb) / (R * T)
    # M in kg/mol, R in J/(mol·K), T in K
    # Let's use M in g/mol, R in J/(mol·K), then convert
    # mdot [kg/s] = (M [g/mol] / 1000 [g/kg]) * Kg [m/s] * Ap [m²] * (Psat - P_amb) [Pa] / (R [J/(mol·K)] * T [K])
    mdot = (m / 1000.0) * kg * ap * (psat - Tambient) / (r_gas * (tp + 273.15))
    return mdot


def compute_flash_and_evaporation(
    chemical: str,
    T_storage: float,
    P_storage: float,
    mass_liquid: float,
    ambient_temperature: float = 25.0,
    ambient_pressure: float = 101325.0,
) -> Dict[str, Optional[float]]:
    """Compute flash evaporation and initial pool spreading for a release.

    Parameters
    ----------
    chemical : str
        Chemical key from CHEMICAL_PROPERTIES dict.
    T_storage : float
        Storage temperature in °C.
    P_storage : float
        Storage pressure in Pa.
    mass_liquid : float
        Initial liquid mass in kg.
    ambient_temperature : float
        Ambient air temperature in °C (default 25°C).
    ambient_pressure : float
        Ambient pressure in Pa (default 101325 Pa).

    Returns
    -------
    Dict[str, Optional[float]]
        Dictionary with flash and pool parameters:
        - 'x_flash': Flash vapor fraction (0-1)
        - 't_bp': Boiling point in °C
        - 'mw': Molecular weight in g/mol
        - 'rho_l': Liquid density at storage condition (estimated) in kg/m³
        - 'p_sat': Saturation pressure at T_storage in Pa
        - 'dotm_evap': Initial evaporation rate in kg/s (estimated)
        - 'none' values for parameters not directly computable without
          additional pool evolution logic.
    """
    props = CHEMICAL_PROPERTIES.get(chemical)
    if props is None:
        raise ValueError(f"Unknown chemical: {chemical}. Available: {list(CHEMICAL_PROPERTIES.keys())}")

    T_bp = props["t_bp"]
    delta_h_vap = props["delta_h_vap"]
    cp_l = props["cp_l"]
    mw = props["mw"]
    rho_l_20 = props["rho_l_20"]

    # 1. Flash vapor fraction
    x_flash = flash_vapor_fraction(T_storage, T_bp, delta_h_vap, cp_l)

    # 2. Estimate liquid pool temperature (assume subcooled)
    T_pool = min(T_storage, T_bp - 1.0)  # stay below bp

    # 3. Saturation pressure at pool temperature using Antoine
    a, b, c = props["p_sat_coeff"], props["p_sat_b"], props["p_sat_c"]
    p_sat_bar = antoine_pvp(T_pool, a, b, c)
    p_sat = p_sat_bar * 100000.0  # bar to Pa

    # 4. Liquid density adjustment (simple linear approx from 20°C)
    # For many hydrocarbons/ammonia: rho_l(T) ≈ rho_l(20) * (1 - beta * (T - 20))
    # We'll use a rough constant for now
    rho_l = rho_l_20  # kg/m³ - will be updated in pool evolution

    # 5. Initial pool radius from mass and density
    # Assuming circular pool, depth ~ 0.1 m for initial spread
    pool_depth = 0.1  # m assumed initial depth
    r_initial = np.sqrt(mass_liquid / (rho_l * pool_depth * np.pi))

    # 6. Initial evaporation rate (using approximate mass transfer coefficient)
    # Kg ~ 0.001 m/s for still air, Ap = pi * r_initial^2
    kg_approx = 0.001  # m/s approximate
    ap = np.pi * r_initial ** 2  # m²
    dotm_evap = evaporation_rate(
        m=mw, kg=kg_approx, ap=ap,
        psat=p_sat, Tambient=ambient_pressure,
        tp=T_pool, r=r_initial,
    )

    result = {
        "x_flash": x_flash,
        "t_bp": T_bp,
        "mw": mw,
        "rho_l": rho_l,
        "p_sat": p_sat,
        "dotm_evap": dotm_evap,
        "r_initial": r_initial,
    }

    return result