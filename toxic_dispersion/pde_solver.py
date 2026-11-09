"""2D Advection-Dispersion PDE solver using explicit and implicit (ADI) schemes."""

import math
import numpy as np
from typing import Dict, Tuple, Optional, List


def solve_advection_dispersion_2d_explicit(
    C0: np.ndarray,
    u: float,
    v: float,
    K_y: float,
    K_z: float,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    source: Optional[np.ndarray] = None,
    boundary: str = "dirichlet",
) -> np.ndarray:
    """Solve 2D advection-dispersion equation using explicit finite differences.

    Uses upwind differencing for advection and central differences for diffusion.
    """
    ny, nx = C0.shape
    if source is None:
        source = np.zeros((ny, nx))
    elif source.shape != (ny, nx):
        raise ValueError(f"source shape {source.shape} must match C0 shape {C0.shape}")

    C = C0.copy().astype(float)

    adv_x = u * dt / dx
    adv_y = v * dt / dy
    diff_y = K_y * dt / (dy ** 2)
    diff_z = K_z * dt / (dy ** 2)

    for _ in range(nt):
        C_new = C.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if u >= 0:
                    adv_x_term = adv_x * (C[i, j] - C[i, j - 1])
                else:
                    adv_x_term = adv_x * (C[i, j + 1] - C[i, j])

                if v >= 0:
                    adv_y_term = adv_y * (C[i, j] - C[i - 1, j])
                else:
                    adv_y_term = adv_y * (C[i + 1, j] - C[i, j])

                diff_term = diff_y * (C[i + 1, j] - 2.0 * C[i, j] + C[i - 1, j]) + \
                            diff_z * (C[i, j + 1] - 2.0 * C[i, j] + C[i, j - 1])

                source_term = source[i, j] * dt
                C_new[i, j] = C[i, j] - adv_x_term - adv_y_term + diff_term + source_term

        if boundary == "dirichlet":
            C_new[0, :] = 0.0
            C_new[ny - 1, :] = 0.0
            C_new[:, 0] = 0.0
            C_new[:, nx - 1] = 0.0
        elif boundary == "zero_flux":
            C_new[0, :] = C[1, :]
            C_new[ny - 1, :] = C[ny - 2, :]
            C_new[:, 0] = C[:, 1]
            C_new[:, nx - 1] = C[:, nx - 2]

        C = C_new

    return C


def solve_advection_dispersion_2d_adi(
    C0: np.ndarray,
    u: float,
    v: float,
    K_y: float,
    K_z: float,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    source: Optional[np.ndarray] = None,
    boundary: str = "dirichlet",
) -> np.ndarray:
    """Solve 2D advection-dispersion equation using ADI (Alternating Direction Implicit).

    The ADI method splits each full time step into two half-steps:
    1. Implicit in x, explicit in y
    2. Explicit in x, implicit in y

    This eliminates the CFL constraint for diffusion, allowing larger dt.
    """
    ny, nx = C0.shape
    if source is None:
        source = np.zeros((ny, nx))
    elif source.shape != (ny, nx):
        raise ValueError(f"source shape {source.shape} must match C0 shape {C0.shape}")

    C = C0.copy().astype(float)

    r_y = K_y * dt / (dy ** 2)
    r_z = K_z * dt / (dy ** 2)
    adv_x = u * dt / (2.0 * dx)
    adv_y = v * dt / (2.0 * dy)

    for _ in range(nt):
        # ADI Step 1: Implicit in X, Explicit in Y
        for j in range(1, nx - 1):
            rhs = np.zeros(ny)
            for i in range(ny):
                src = source[i, j] * dt / 2.0 if source is not None else 0.0
                if v >= 0 and i > 0:
                    adv_y_term = adv_y * (C[i, j] - C[i - 1, j])
                elif v < 0 and i < ny - 1:
                    adv_y_term = adv_y * (C[i + 1, j] - C[i, j])
                else:
                    adv_y_term = 0.0
                rhs[i] = C[i, j] - adv_y_term + src

            if boundary == "dirichlet":
                rhs[0] = rhs[0]  # C_0 = 0 already
                rhs[ny - 1] = rhs[ny - 1]  # C_{ny-1} = 0 already

            # Solve tridiagonal: (1+2r)*C_i - r*C_{i-1} - r*C_{i+1} = rhs_i
            C[:, j] = _thomas_solve(n=ny, a=np.full(ny - 1, -r_y), b=np.full(ny, 1.0 + 2.0 * r_y),
                                     c=np.full(ny - 1, -r_y), d=rhs)

        # ADI Step 2: Explicit in X, Implicit in Y
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                laplacian_y = (C[i + 1, j] - 2.0 * C[i, j] + C[i - 1, j])

                if u >= 0:
                    adv_x_term = adv_x * (C[i, j] - C[i, j - 1])
                else:
                    adv_x_term = adv_x * (C[i, j + 1] - C[i, j])

                C[i, j] = (C[i, j] - adv_x_term +
                           K_y * dt / (dy ** 2) * laplacian_y +
                           K_z * dt / (dy ** 2) * (C[i, j + 1] - 2.0 * C[i, j] + C[i, j - 1]) +
                           source[i, j] * dt) / (1.0 + 1e-12)

        if boundary == "dirichlet":
            C[0, :] = 0.0
            C[ny - 1, :] = 0.0
            C[:, 0] = 0.0
            C[:, nx - 1] = 0.0
        elif boundary == "zero_flux":
            C[0, :] = C[1, :]
            C[ny - 1, :] = C[ny - 2, :]
            C[:, 0] = C[:, 1]
            C[:, nx - 1] = C[:, nx - 2]

    return C


def _thomas_solve(n: int, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal system: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]"""
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([d[0] / b[0] if b[0] != 0 else 0.0])

    cp = np.zeros(n - 1)
    dp = np.zeros(n)

    if abs(b[0]) > 1e-15:
        dp[0] = d[0] / b[0]
        cp[0] = c[0] / b[0] if n > 1 else 0.0

    for i in range(1, n):
        denom = b[i] - a[i - 1] * cp[i - 1] if i - 1 < len(cp) else b[i]
        if abs(denom) > 1e-15:
            dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
            if i < n - 1:
                cp[i] = c[i] / denom
        else:
            dp[i] = 0.0
            if i < n - 1:
                cp[i] = 0.0

    x = np.zeros(n)
    x[n - 1] = dp[n - 1] if abs(b[n - 1]) > 1e-15 else 0.0

    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1] if i < len(cp) else dp[i]

    return x


def plume_centerline_profile(
    x: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    centerline_idx: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract centerline and crosswind profiles from 2D concentration field."""
    if centerline_idx is None:
        centerline_idx = int(np.argmax(np.sum(C, axis=1)))

    centerline_idx = max(0, min(centerline_idx, C.shape[0] - 1))
    centerline = C[centerline_idx, :]

    cy_start = max(1, C.shape[1] // 4)
    cy_end = min(C.shape[1] - 1, C.shape[1] * 3 // 4)
    crosswind_profile = np.mean(C[centerline_idx, cy_start:cy_end], axis=0)

    return centerline, crosswind_profile


def find_mortality_radii(
    x: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    probit_params: Dict[str, float],
    ergp_thresholds: Dict[str, float],
) -> Dict[str, float]:
    """Find downwind distances where probit fatality probability exceeds ERPG thresholds."""
    k1 = probit_params["k1"]
    k2 = probit_params["k2"]

    results = {}
    C_floor = np.maximum(C, 1e-10)
    Y_field = k1 + k2 * np.log(C_floor)
    sqrt2 = math.sqrt(2.0)
    Pf_field = 0.5 * (1.0 + math.erf((Y_field - 5.0) / sqrt2))
    Pf_field = np.clip(Pf_field, 0.0, 1.0)

    for level, threshold in ergp_thresholds.items():
        exceedance = C >= threshold
        if not np.any(exceedance):
            results[level] = 0.0
            continue

        max_dist = 0.0
        for i in range(exceedance.shape[0]):
            for j in range(exceedance.shape[1]):
                if exceedance[i, j]:
                    max_dist = max(max_dist, float(x[i]))
        results[level] = max_dist

    return results