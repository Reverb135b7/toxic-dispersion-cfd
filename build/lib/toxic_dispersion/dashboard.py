"""Streamlit interactive hazard dashboard (Phase 4).

Provides UI controls for wind speed, stability class, release diameter,
chemical selector, and displays dynamic downwind plume footprint and
lethality radius with ERPG contour overlays.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .source_term import compute_flash_and_evaporation
from .dispersion import (
    gravity_slumping_velocity,
    pasquill_gifford_sigma_v2,
    gaussian_plume_full_grid,
)
from .probit import compute_fatality
from .geospatial import (
    concentration_to_geojson,
    plot_erpggu_contours,
    compute_exceedance_fractions,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Toxic Release Hazard Dashboard",
    page_icon="☢️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Chemical property defaults (from Phase 1 source_term module)
# ---------------------------------------------------------------------------
CHEMICAL_DEFAULTS = {
    "ammonia": {
        "mw": 17.031,
        "t_bp": -33.34,
        "delta_h_vap": 23.35e3,
        "cp_l": 4.7,
        "rho_l_20": 681.0,
        "p_sat_coeff": 61.7,
        "p_sat_b": 13.28,
        "p_sat_c": -5.2,
    },
    "chlorine": {
        "mw": 70.906,
        "t_bp": -34.05,
        "delta_h_vap": 20.28e3,
        "cp_l": 0.48,
        "rho_l_20": 1639.0,
        "p_sat_coeff": 12.0,
        "p_sat_b": 2.75,
        "p_sat_c": -3.8,
    },
}


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def sidebar_controls() -> Dict:
    """Collect all UI controls from the sidebar.

    Returns
    -------
    Dict
        Dictionary of all user-selected parameters.
    """
    st.sidebar.title("Release Parameters")

    # Chemical selection
    chemical = st.sidebar.selectbox(
        "Toxic Chemical",
        options=["ammonia", "chlorine"],
        index=0,
        help="Select the toxic chemical for dispersion modeling",
    )

    # Release parameters
    st.sidebar.subheader("Release Conditions")
    release_diameter = st.sidebar.slider(
        "Release Diameter (m)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Effective release opening diameter",
    )

    release_pressure = st.sidebar.number_input(
        "Storage Pressure (bar)",
        min_value=1.0,
        max_value=500.0,
        value=50.0,
        step=1.0,
        help="Initial storage tank pressure",
    )

    release_temperature = st.sidebar.number_input(
        "Storage Temperature (°C)",
        min_value=-50.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
        help="Initial liquid temperature",
    )

    # Dispersion parameters
    st.sidebar.subheader("Dispersion Conditions")
    wind_speed = st.sidebar.slider(
        "Wind Speed (m/s)",
        min_value=0.5,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Mean wind speed at release height",
    )

    stability_class = st.sidebar.selectbox(
        "Pasquill-Gifford Stability Class",
        options=["A", "B", "C", "D", "E", "F"],
        index=3,  # D (neutral)
        help="Atmospheric stability class affecting dispersion rates",
    )

    release_height = st.sidebar.slider(
        "Release Height (m)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Effective release height above ground",
    )

    exposure_duration = st.sidebar.slider(
        "Exposure Duration (seconds)",
        min_value=1,
        max_value=600,
        value=60,
        step=1,
        help="Duration of toxic exposure for probit calculation",
    )

    pool_depth = st.sidebar.slider(
        "Initial Pool Depth (m)",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Assumed initial pool depth for radius calculation",
    )

    return {
        "chemical": chemical,
        "release_diameter": release_diameter,
        "release_pressure": release_pressure,
        "release_temperature": release_temperature,
        "wind_speed": wind_speed,
        "stability_class": stability_class,
        "release_height": release_height,
        "exposure_duration": exposure_duration,
        "pool_depth": pool_depth,
    }


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------
def compute_source_term(params: Dict) -> Dict:
    """Compute flash evaporation and pool spreading (Phase 1).

    Parameters
    ----------
    params : Dict
        Parameters from sidebar_controls().

    Returns
    -------
    Dict
        Source-term results including flash fraction, evaporation rate, etc.
    """
    chemical = params["chemical"]
    T_storage = params["release_temperature"]
    P_storage = params["release_pressure"] * 100000.0  # bar to Pa
    mass_liquid = 1000.0  # assume 1000 kg initial liquid mass for demo

    result = compute_flash_and_evaporation(
        chemical=chemical,
        T_storage=T_storage,
        P_storage=P_storage,
        mass_liquid=mass_liquid,
    )

    return result


def compute_dispersion(params: Dict) -> Dict:
    """Compute Gaussian plume dispersion and concentration field (Phase 2-3).

    Parameters
    ----------
    params : Dict
        Parameters from sidebar_controls().

    Returns
    -------
    Dict
        Concentration field and derived parameters.
    """
    chemical = params["chemical"]
    u = params["wind_speed"]
    stability = params["stability_class"]
    h = params["release_height"]
    mw = CHEMICAL_DEFAULTS[chemical]["mw"]

    # Compute dispersion coefficients at various downwind distances
    # Use a range of downwind distances from 10m to 10000m
    x_distances = np.logspace(1, 4, 100)  # 10m to 10km

    sy, sz = pasquill_gifford_sigma_v2(x_distances, stability)

    # Compute ground-level centerline concentration
    # Assume unit emission rate for demonstration; will scale later
    q = 1.0  # kg/s emission rate (will be derived from source term)

    centerline_conc = gaussian_plume_centerline(
        x=x_distances,
        q=q,
        u=u,
        sy=sy,
        sz=sz,
        h=h,
        ground_reflection=True,
    )

    # Compute full 2D concentration grid
    y_distances = np.linspace(-200, 200, 200)  # crosswind extent
    conc_grid = gaussian_plume_full_grid(
        x=x_distances,
        y=y_distances,
        q=q,
        u=u,
        sy=sy[-1],  # use value at max distance
        sz=sz[-1],
        h=h,
        ground_reflection=True,
    )

    # Compute gravity slumping velocity for dense-gas initial phase
    # Need gas density - estimate from ideal gas law
    # rho_g = P * M / (R * T) for the vapor
    T_kelvin = params["release_temperature"] + 273.15
    # For simplicity, use approximate gas density
    rho_g = (params["release_pressure"] * 100000.0 * mw) / (8314.462 * T_kelvin)

    # Air density at ambient conditions
    rho_a = 1.225  # kg/m³ at 15°C, 1 atm

    # Gravity slumping velocity
    ug = gravity_slumping_velocity(
        eg=0.65,  # typical entrainment coefficient
        g=9.81,
        rhog=rho_g,
        rhoa=rho_a,
        hcloud=params["release_height"],
    )

    # Compute toxic dose and probit fatality
    # Use centerline concentration integrated over exposure duration
    # Simple approximation: D = C * t_exp (for n=1)
    centerline_max = np.max(centerline_conc)
    dosage = compute_toxic_dose(
        concentration_time_series=np.full_like(x_distances, centerline_max),
        exposure_time=params["exposure_duration"],
        n_order=1.0,
    )

    fatality_result = compute_fatality(
        dosage=dosage["dosage"],
        chemical=chemical,
    )

    return {
        "x_distances": x_distances,
        "sy": sy,
        "sz": sz,
        "centerline_conc": centerline_conc,
        "conc_grid": conc_grid,
        "y_distances": y_distances,
        "ug": ug,
        "rho_g": rho_g,
        "dosage_result": fatality_result,
        "centerline_max": centerline_max,
    }


# ---------------------------------------------------------------------------
# Main dashboard layout
# ---------------------------------------------------------------------------
def main():
    """Main Streamlit dashboard entry point."""
    st.title("☢️ Toxic Industrial Release Hazard Dashboard")
    st.caption("Dynamic CFD & Heavy-Gas Dispersion Modeling")

    # Get sidebar controls
    params = sidebar_controls()

    # Compute source term (Phase 1)
    with st.spinner("Computing source-term & flash evaporation..."):
        source_result = compute_source_term(params)

    # Compute dispersion (Phase 2-3)
    with st.spinner("Computing dispersion & concentration field..."):
        disp_result = compute_dispersion(params)

    # Display results in columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Downwind Concentration Profile")
        fig = go.Figure()

        # Centerline concentration curve
        x = disp_result["x_distances"]
        conc = disp_result["centerline_conc"]

        fig.add_trace(go.Scatter(
            x=x,
            y=conc,
            mode="lines",
            name="Centerline Concentration",
            line=dict(color="red", width=2),
        ))

        # ERPG thresholds as horizontal lines
        chemical = params["chemical"]
        from .geospatial import ERPG_THRESHOLDS
        thresholds = ERPG_THRESHOLDS.get(chemical, {})

        for level, threshold in thresholds.items():
            # Convert threshold to appropriate units for display
            # Rough approximation: show as same scale as centerline
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"ERPG-{level[-1]}",
                annotation_position="right",
            )

        fig.update_layout(
            xaxis_title="Downwind Distance (m)",
            yaxis_title="Concentration (kg/m³)",
            hovermode="x",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Metrics")

        # Flash vapor fraction
        st.metric(
            "Flash Vapor Fraction",
            f"{source_result['x_flash']:.3f}",
            help="Fraction of liquid that flashes to vapor upon release",
        )

        # Gravity slumping velocity
        st.metric(
            "Slumping Velocity",
            f"{disp_result['ug']:.2f} m/s",
            help="Initial gravity-driven cloud spread speed",
        )

        # Lethality radius (distance where P_fatality > 0.5)
        prob = disp_result["dosage_result"]["fatality_probability"]
        st.metric(
            "Centerline Fatality Probability",
            f"{prob:.3f}",
            help="Probability of fatality at centerline for exposure duration",
        )

        # Lethality radius estimation
        # Find distance where concentration drops to ERPG-3 level
        thresholds = ERPG_THRESHOLDS.get(chemical, {})
        if thresholds:
            ergp3 = list(thresholds.values())[2]  # ergp_3 value
            # Simple inversion for distance (approximate)
            conc_at_ergp3 = ergp3
            # Find approximate distance
            mask = disp_result["centerline_conc"] >= conc_at_ergp3
            if np.any(mask):
                idx = np.argmax(mask)  # first index where condition met
                # Actually find last index where still above
                above = disp_result["centerline_conc"] >= conc_at_ergp3
                if np.any(above):
                    last_above = np.where(above)[0][-1]
                    lethality_radius = x[last_above]
                    st.metric(
                        "ERPG-3 Lethality Radius",
                        f"{lethality_radius:.0f} m",
                        help="Distance where ERPG-3 threshold is exceeded",
                    )

        # Dosage and probit details
        st.subheader("Probit Analysis")
        fd = disp_result["dosage_result"]
        st.metric("Dosage D", f"{fd['dosage']:.4e}")
        st.metric("Probit Score Y", f"{fd['probit_score']:.2f}")
        st.metric("Fatality Probability", f"{fd['fatality_probability']:.3f}")

    # Full concentration heatmap
    st.subheader("2D Concentration Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_erpggu_contours(
        ax=ax2,
        x=disp_result["x_distances"],
        y=disp_result["y_distances"],
        concentration=disp_result["conc_grid"],
        chemical=chemical,
        erpg_thresholds=ERPG_THRESHOLDS.get(chemical, {}),
        add_colorbar=True,
    )
    ax2.set_title("ERPG Threshold Contours")
    st.pyplot(fig2)

    # Exceedance fractions
    st.subheader("Exceedance Fractions")
    exc = compute_exceedance_fractions(
        x=disp_result["x_distances"],
        y=disp_result["y_distances"],
        concentration=disp_result["conc_grid"],
        chemical=chemical,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("ERPG-1 Exceedance", f"{exc['ergp_1']:.1%}")
    with col_b:
        st.metric("ERPG-2 Exceedance", f"{exc['ergp_2']:.1%}")
    with col_c:
        st.metric("ERPG-3 Exceedance", f"{exc['ergp_3']:.1%}")


if __name__ == "__main__":
    main()