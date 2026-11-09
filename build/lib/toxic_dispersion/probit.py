"""Toxic dose and Eisenberg probit fatality module (Phase 2).

Converts concentration-time integration to probit mortality scores
per Eisenberg's methodology for chlorine and ammonia.
"""
import numpy as np
from scipy import integrate
from typing import Dict, Tuple, Optional, Literal


# ---------------------------------------------------------------------------
# Eisenberg probit equation parameters by chemical.
# These are the slope (k2) and intercept (k1) parameters from:
#   Y = k1 + k2 * ln(D)
#   P_fatality = Phi((Y - 5) / 1)  i.e., standard normal CDF at Y-5
# ---------------------------------------------------------------------------

PROBIT_PARAMS: Dict[str, Dict[str, float]] = {
    "ammonia": {
        "k1": -0.98,
        "k2": 2.62,
        "description": "Eisenberg probit parameters for ammonia (Y = -0.98 + 2.62*ln(D))",
    },
    "chlorine": {
        "k1": -1.04,
        "k2": 3.02,
        "description": "Eisenberg probit parameters for chlorine (Y = -1.04 + 3.02*ln(D))",
    },
    "hydrogen_sulfide": {
        "k1": -0.85,
        "k2": 2.45,
        "description": "Eisenberg probit parameters for H2S",
    },
}


def compute_toxic_dose(
    concentration_time_series: np.ndarray,
    exposure_time: float,
    n_order: float = 1.0,
) -> float:
    """Compute toxic dosage D = ∫ C(t)^n dt from 0 totexp.

    The dosage integration raises concentration to power n before integrating.
    For n=1, this is the linear dosage C*t integration.

    Parameters
    ----------
    concentration_time_series : np.ndarray
        Concentration C(t) in kg/m³ or mg/m³ (consistent throughout).
    exposure_time : float
        Exposure duration in seconds.
    n_order : float, optional
        Power to which concentration is raised (default 1.0).

    Returns
    -------
    float
        Toxic dosage D in (concentration^n * time) units.
    """
    t = np.linspace(0, exposure_time, len(concentration_time_series))

    # Ensure concentration array matches time array
    if len(concentration_time_series) != len(t):
        raise ValueError(
            f"concentration_time_series length ({len(concentration_time_series)}) "
            f"must match exposure_time discretization"
        )

    # Compute integrand: C(t)^n
    integrand = concentration_time_series ** n_order

    # Use trapezoidal integration
    dosage, _ = integrate.trapz(integrand, t)
    return float(dosage)


def probit_score(
    dosage: float,
    k1: float,
    k2: float,
) -> float """Compute Eisenberg probit score Y = k1 + k2 * ln(D).

    Parameters
    ----------
    dosage : float
        Toxic dosage D.
    k1 : float
        Probit intercept parameter.
    k2 : float
        Probit slope parameter.

    Returns
    -------
    float
        Probit score Y.
    """
    if dosage <= 0:
        return float("-inf") if k2 > 0 else float("inf")

    y = k1 + k2 * np.log(dosage)
    return float(y)


def fatality_probability(
    y: float,
) -> float: """Compute fatality probability from probit score.

    P_fatality = Phi(Y - 5) = (1 / sqrt(2*pi)) * integral from -inf to Y-5 of exp(-u^2/2) du

    Parameters
    ----------
    y : float
        Probit score Y.

    Returns
    -------
    float
        Fatality probability in [0, 1].
    """
    # Standard normal CDF at (Y - 5)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    z = y - 5.0
    pf = 0.5 * (1.0 + erf_safe(z))
    return float(np.clip(pf, 0.0, 1.0))


def erf_safe(x: float) -> float: """Complementary error function approximation.

    Uses math.erf for numerical stability.
    """
    return np.math.erf(x)


def compute_fatality(
    dosage: float,
    chemical: str = "chlorine",
) -> Dict[str, float]:
    """Compute probit score and fatality probability for a given dosage.

    Parameters
    ----------
    dosage : float
        Toxic dosage D in appropriate units (typically mg·s/m³ or ppm·s).
    chemical : str, optional
        Chemical key from PROBIT_PARAMS dict (default "chlorine").

    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - 'dosage': The input dosage D
        - 'probit_score': Y = k1 + k2*ln(D)
        - 'fatality_probability': P_fatality in [0, 1]
        - 'k1': intercept parameter
        - 'k2': slope parameter
    """
    params = PROBIT_PARAMS.get(chemical)
    if params is None:
        raise ValueError(
            f"Unknown chemical: {chemical}. "
            f"Available: {list(PROBIT_PARAMS.keys())}"
        )

    k1 = params["k1"]
    k2 = params["k2"]

    y = probit_score(dosage, k1, k2)
    pf = fatality_probability(y)

    return {
        "dosage": dosage,
        "probit_score": y,
        "fatality_probability": pf,
        "k1": k1,
        "k2": k2,
    }