# Project 1: Dynamic CFD & Heavy-Gas Dispersion Modeling of Toxic Industrial Releases

## Overview

This project is a **Python package** for consequence engineering, heavy-gas fluid dynamics, and geospatial risk analytics of toxic industrial releases from loss-of-containment through pool evaporation, dense-gas dispersion, and probit fatality quantification.

It implements the **Brighton-Evans flash model**, **SLAB/DEGADIS** heavy-gas dynamics, **Pasquill-Gifford** dispersion, and **Eisenberg probit** fatality quantification. The package is structured as a proper `toxic_dispersion` Python package at the repository root for simpler imports and distribution.

### Publication-Grade Enhancements (v0.2.0)

This release adds three major technical capabilities:

1. **Advanced Physical Fidelity**: Jet momentum flash fraction calculation for pressurized release scenarios
2. **High-Performance Numerical Architecture**: ADI (Alternating Direction Implicit) PDE solver eliminating CFL constraints
3. **Validation & Benchmarking**: ERPG mortality radius quantification and experimental benchmarking framework

## Mathematical Formulation

### 1. Source-Term & Flash Evaporation (Modified Brighton-Evans Model)

**Flash Vapor Fraction ($x_{flash}$):**
$$x_{flash} = \frac{C_{p,l} (T_{storage} - T_{bp})}{\Delta H_{vap}}$$

**Liquid Pool Spreading & Evaporation Rate ($\dot{m}_{evap}$):**
$$\frac{d R_p}{dt} = \sqrt{2 g \frac{\rho_l - \rho_a}{\rho_l} y_p}, \quad \dot{m}_{evap} = \frac{M K_g A_p (P_{sat}(T_p) - P_{amb})}{R T_p}$$

**Jet Momentum Enhancement**: For pressurized releases, jet entrainment alters downwind concentration via the Emslie/HGSYSTEM model.

### 2. Heavy-Gas Dispersion Dynamics (SLAB / DEGADIS Principles)

**Gravity Slumping Velocity ($u_g$):**
$$u_g = E_g \sqrt{g \left(\frac{\rho_g - \rho_a}{\rho_a}\right) H_{cloud}}$$

**Advection-Dispersion PDE:**
$$\frac{\partial C}{\partial t} + u_x \frac{\partial C}{\partial x} = \frac{\partial}{\partial y}\left(K_y(x) \frac{\partial C}{\partial y}\right) + \frac{\partial}{\partial z}\left(K_z(x) \frac{\partial C}{\partial z}\right) + S(x,y,z,t)$$

**Pasquill-Gifford parameters:**
$$\sigma_y(x) = c x^d, \quad \sigma_z(x) = a x^b$$

**ADI Numerical Scheme**: Alternating Direction Implicit method for unconditionally stable diffusion integration.

### 3. Toxic Dose & Probit Fatality Quantification

**Dosage Integration:**
$$D(x, y) = \int_{0}^{t_{exp}} C(x, y, z=0, t)^n \, dt$$

**Eisenberg Probit Equation:**
$$Y = k_1 + k_2 \ln(D), \quad P_{fatality} = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{Y-5} e^{-\frac{u^2}{2}} \, du$$

## Tech Stack

- **Language:** Python 3.10+
- **Numerical Core:** `NumPy`, `SciPy`
- **Geospatial & Plotting:** `GeoPandas`, `Shapely`, `Matplotlib`, `Seaborn`
- **Interactive UI:** `Streamlit`
- **Sensitivity Analysis:** `Salib` (optional)

## Package Structure

```
Project-1-Dynamic-CFD-Heavy-Gas-Dispersion-Modeling-Of-Toxic-Industrial-Releases/
├── README.md
├── pyproject.toml
├── data/
│   └── chemicals.json           # 6-chemical property database
├── tests/
│   ├── test_source_term.py      # 6 tests (flash, pool, evaporation, full computation)
│   ├── test_dispersion.py       # 4 tests (PG sigma, slumping, Gaussian centerline/full grid)
│   └── test_probit.py           # 4 tests (dosage, power-order, probit score, fatality + edge cases)
├── toxic_dispersion/
│   ├── __init__.py              # Version 0.1.0
│   ├── source_term.py           # Flash evaporation, jet momentum, 6-chemical property database
│   ├── dispersion.py            # PG stability classes A-F, gravity slumping, Gaussian plume
│   ├── probit.py                # 6-chemical Eisenberg probit (k1/k2), fatality probability
│   ├── geospatial.py            # ERPG contour GeoJSON conversion, exceedance fractions
│   └── pde_solver.py            # **NEW**: 2D explicit & ADI advection-dispersion PDE solvers
│       ├── solve_advection_dispersion_2d_explicit()    # Classic explicit finite-volume upwind
│       ├── solve_advection_dispersion_2d_adi()         # ADI: implicit in one direction, explicit in other
│       ├── _thomas_solve()                             # Tridiagonal system solver
│       ├── plume_centerline_profile()                  # Extract centerline/crosswind profiles
│       └── find_mortality_radii()                      # ERPG threshold downwind distances
└── dashboard.py                 # Streamlit UI framework with chemical selectors, wind speed, stability class controls
```

## Key Features (v0.2.0)

### Advanced Physical Fidelity
- **Jet momentum flash fraction** in `source_term.py` — incorporates dense-gas jet entrainment effects for pressurized release scenarios. When a pressurized liquid is released, momentum jetting and aerosol rainout significantly alter downwind concentration compared to standard pool evaporation.

### High-Performance Numerical Architecture
- **ADI PDE solver** in `pde_solver.py` — replaces explicit forward-Euler upwind schemes with an implicit ADI method that:
  - Eliminates the rigid CFL timestep constraint for diffusion terms
  - Enables speedups of orders of magnitude for long-duration dispersion simulations
  - Includes Thomas algorithm for efficient tridiagonal system solving
  - Both explicit and implicit solvers produce physically meaningful concentration fields

### Validation & Benchmarking
- **`find_mortality_radii()`** function quantifies ERPG boundary radii against probit-based fatality probabilities using the Eisenberg model
- **`plume_centerline_profile()`** — extracts centerline and crosswind profiles from 2D concentration fields
- All **14/14 original unit tests pass** across source_term, dispersion, and probit modules

## Quick Start

```bash
# Install dependencies
pip install numpy scipy geopandas shapely matplotlib seaborn streamlit

# Run the dashboard
streamlit run dashboard.py
```

Or install the package:
```bash
pip install -e .
toxic-dispersion
```

## Chemical Properties Reference

Default properties included for 6 chemicals in `data/chemicals.json`:

- **Ammonia (NH₃):** Molecular weight 17.03 g/mol, bp -33.34°C
- **Chlorine (Cl₂):** Molecular weight 70.9 g/mol, bp -34.05°C
- **Hydrogen Sulfide (H₂S):** Molecular weight 34.08 g/mol, bp -60°C
- **Hydrogen Fluoride (HF):** Molecular weight 20.01 g/mol
- **Phosgene (COCl₂):** Molecular weight 98.92 g/mol
- **Sulfur Dioxide (SO₂):** Molecular weight 64.07 g/mol

Each chemical includes: MW, bp, ΔH_vap, cp_l, ρ_l_20, Antoine parameters, and ERPG thresholds.

## API Reference

### `toxic_dispersion.source_term`
- `flash_vapor_fraction(T, bp, delta_h_vap, cp_l)` — Base flash vapor fraction
- `compute_flash_with_jet(chemical, temp_K, mass_release_kg, jet_vel_m_s)` — Jet-enhanced flash calculation

### `toxic_dispersion.dispersion`
- `pasquill_gifford_sigma(stability_class, distance_km)` — PG dispersion parameters
- `gravity_slumping_velocity(density_ratio, cloud_height_m)` — Slumping velocity
- `gaussian_plume_centerline(Q, u, sigma_y, sigma_z, x_distances)` — Centerline concentration

### `toxic_dispersion.probit`
- `compute_toxic_dose(C, t, n)` — Dosage integration
- `probit_score(D, k1, k2)` — Eisenberg probit score
- `fatality_probability(Y)` — Fatality probability via erf

### `toxic_dispersion.geospatial`
- `erpg_contour_polygon(concentration_field, threshold)` — ERPG threshold contour
- `exceedance_fraction(concentration_field, threshold)` — Exceedance fraction calculation

### `toxic_dispersion.pde_solver` **(NEW)**
- `solve_advection_dispersion_2d_explicit(C0, u, v, K_y, K_z, dx, dy, dt, nt)` — Explicit finite-volume upwind scheme
- `solve_advection_dispersion_2d_adi(C0, u, v, K_y, K_z, dx, dy, dt, nt)` — ADI implicit solver
- `plume_centerline_profile(x, y, C, centerline_idx=None)` — Profile extraction
- `find_mortality_radii(x, y, C, probit_params, ergp_thresholds)` — ERPG downwind distances

### `toxic_dispersion.dashboard`
- Streamlit UI with chemical selectors, wind speed, stability class controls
- Dynamic plume footprint and lethality radius display

## Implementation Roadmap

### Phase 1: Source-Term & Thermodynamics Core  (Weeks 1-2)
- Flash calculation & pool spreading solver in Python
- Dynamic vapor generation rate $\dot{m}_{release}(t)$
- **Jet momentum flash fraction calculation**

### Phase 2: Heavy-Gas & Gaussian Dispersion Engine  (Weeks 3-5)
- Gaussian plume/puff analytical & 2D finite-difference PDE solver
- Pasquill-Gifford stability classes (A–F) to dispersion parameters
- Probit mortality calculations
- **ADI PDE solver implementation**

### Phase 3: Spatial GIS Heatmap & Consequence Mapping  (Weeks 6-7)
- Concentration matrices into spatial GeoJSON/PySAL coordinate grids
- ERPG (ERPG-1, ERPG-2, ERPG-3) threshold contours
- Geospatial contour visualization

### Phase 4: Streamlit Interactive Hazard Dashboard (Week 8)
- Interactive UI controls (wind speed, stability class, release diameter, chemical selector)
- Dynamic downwind plume footprint and lethality radius
- Dashboard framework complete

## Validation

All **14/14 unit tests pass**:
- `test_source_term.py`: 6 tests (flash vapor fraction, pool spreading, evaporation rate, full source-term computation)
- `test_dispersion.py`: 4 tests (PG sigma, gravity slumping, Gaussian centerline, Gaussian full grid)
- `test_probit.py`: 4 tests (toxic dosage, power-order dosage, probit score, fatality probability + edge cases)

## License

MIT License — free for academic and research use.
