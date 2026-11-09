"""GIS heatmap and consequence mapping module (Phase 3).

Parses concentration matrices into spatial GeoJSON/PySAL coordinate grids
and plots ERPG (Emergency Response Planning Guidelines) threshold contours.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from typing import Dict, Tuple, Optional, List
import json


# ---------------------------------------------------------------------------
# ERPG (Emergency Response Planning Guidelines) threshold values.
# These are approximate values in mg/m³ or ppm depending on chemical.
# ERPG-1: No irritation or reversible health effects expected.
# ERPG-2: Irritation or other health effects are unlikely.
# ERPG-3: Death or serious injury would likely occur.
# ---------------------------------------------------------------------------

ERPG_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ammonia": {
        "ergp_1": 25.0,   # mg/m³  - approved for 60 min
        "ergp_2": 150.0,  # mg/m³  - approved for 60 min
        "ergp_3": 3400.0, # mg/m³  - approved for 30 min
    },
    "chlorine": {
        "ergp_1": 0.5,    # ppm  (≈ 1.3 mg/m³ at 25°C, 1 atm)
        "ergp_2": 2.0,    # ppm  (≈ 5.1 mg/m³)
        "ergp_3": 10.0,   # ppm  (≈ 25.4 mg/m³)
    },
    "hydrogen_sulfide": {
        "ergp_1": 10.0,   # ppm  (≈ 14.8 mg/m³)
        "ergp_2": 50.0,   # ppm  (≈ 74.0 mg/m³)
        "ergp_3": 100.0,  # ppm  (≈ 148.0 mg/m³)
    },
}


def concentration_to_geojson(
    x: np.ndarray,
    y: np.ndarray,
    concentration: np.ndarray,
    chemical: str,
    erpg_thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """Convert concentration matrix to GeoJSON with ERPG contour polygons.

    Creates polygon features for each ERPG threshold level where
    concentration exceeds the threshold.

    Parameters
    ----------
    x : np.ndarray
        Downwind distance array in meters (1D).
    y : np.ndarray
        Crosswind distance array in meters (1D).
    concentration : np.ndarray
        2D concentration array [len(x), len(y)] in appropriate units.
    chemical : str
        Chemical key for ERPG threshold lookup.
    erpg_thresholds : Dict[str, float], optional
        Custom ERPG thresholds {ergp_1, ergp_2, ergp_3}. If None, uses
        defaults from ERPG_THRESHOLDS.

    Returns
    -------
    Dict
        GeoJSON FeatureCollection with polygons for each ERPG level.
    """
    if erpg_thresholds is None:
        erpg_thresholds = ERPG_THRESHOLDS.get(chemical, {})

    if not erpg_thresholds:
        raise ValueError(f"No ERPG thresholds defined for chemical: {chemical}")

    # Ensure 2D concentration array
    concentration = np.asarray(concentration, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if concentration.ndim != 2:
        raise ValueError("concentration must be a 2D array")

    if concentration.shape[0] != len(x) or concentration.shape[1] != len(y):
        raise ValueError(
            f"concentration shape {concentration.shape} must match "
            f"x length {len(x)} and y length {len(y)}"
        )

    # Create feature collection
    features = []

    for ergp_level, threshold in erpg_thresholds.items():
        # Create mask where concentration exceeds threshold
        mask = concentration >= threshold

        if not np.any(mask):
            # No area exceeds this threshold; optionally add a note
            continue

        # Find contour points using simple polygonization
        # Build grid of points where concentration >= threshold
        xs, ys = np.meshgrid(x, y, indexing="ij")
        points = []

        for i in range(len(x)):
            for j in range(len(y)):
                if mask[i, j]:
                    points.append(Point(float(xs[i, j]), float(ys[i, j])))

        if len(points) < 3:
            # Not enough points for a polygon
            continue

        # Create convex hull as simplified polygon
        # In production, would use proper contour tracing (marching squares)
        try:
            hull = Polygon().from_coordinates(
                [(p.x, p.y) for p in points]
            )
            if hull.is_valid and hull.area > 0:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "ergp_level": ergp_level,
                        "threshold_mg_m3": threshold,
                        "chemical": chemical,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(hull.exterior.coords)],
                    },
                })
        except Exception:
            # Skip if polygon creation fails
            continue

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    return feature_collection


def plot_erpggu_contours(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    concentration: np.ndarray,
    chemical: str,
    erpg_thresholds: Optional[Dict[str, float]] = None,
    add_colorbar: bool = True,
) -> None:
    """Plot ERPG threshold contours on a given matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    x : np.ndarray
        Downwind distance array in meters.
    y : np.ndarray
        Crosswind distance array in meters.
    concentration : np.ndarray
        2D concentration array [len(x), len(y)].
    chemical : str
        Chemical key for ERPG threshold lookup.
    erpg_thresholds : Dict[str, float], optional
        Custom ERPG thresholds. If None, uses defaults.
    add_colorbar : bool, optional
        Whether to add a colorbar (default True).
    """
    import matplotlib.pyplot as plt

    if erpg_thresholds is None:
        erpg_thresholds = ERPG_THRESHOLDS.get(chemical, {})

    # Plot concentration as heatmap
    cax = ax.pcolormesh(x, y, concentration, shading="auto", cmap="Reds", alpha=0.6)

    if add_colorbar:
        plt.colorbar(cax, ax=ax, label="Concentration (kg/m³)")

    # Overlay ERPG contour polygons
    geo_json = concentration_to_geojson(
        x, y, concentration, chemical, erpg_thresholds
    )

    if geo_json["features"]:
        # Plot each ERPG polygon
        gdf = gpd.GeoDataFrame.from_features(geo_json)
        gdf.plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=2)

        # Add threshold labels
        for ergp_level in ["ergp_1", "ergp_2", "ergp_3"]:
            if ergp_level in erpg_thresholds:
                threshold = erpg_thresholds[ergp_level]
                # Find approximate center of exceedance area
                mask = concentration >= threshold
                if np.any(mask):
                    # Centroid of exceedance region
                    cy, cx = np.where(mask)
                    if len(cx) > 0:
                        x_center = np.mean(x[cx])
                        y_center = np.mean(y[cy])
                        ax.text(
                            x_center,
                            y_center,
                            f"ERPG-{ergp_level[-1]}",
                            color="blue",
                            fontweight="bold",
                            fontsize=10,
                            bbox=dict(facecolor="white", alpha=0.8),
                        )


def compute_exceedance_fractions(
    x: np.ndarray,
    y: np.ndarray,
    concentration: np.ndarray,
    chemical: str,
) -> Dict[str, float]:
    """Compute the fraction of the modeled area that exceeds
    each ERPG threshold.

    Parameters
    ----------
    x : np.ndarray
        Downwind distance array in meters.
    y : np.ndarray
        Crosswind distance array in meters.
    concentration : np.ndarray
        2D concentration array.
    chemical : str
        Chemical key for ERPG thresholds.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping ergp_1, ergp_2, ergp_3 to the fraction of grid
        cells exceeding that threshold (0.0 to 1.0).
    """
    erpg_thresholds = ERPG_THRESHOLDS.get(chemical, {})
    if not erpg_thresholds:
        raise ValueError(f"No ERPG thresholds for {chemical}")

    total_cells = concentration.size
    if total_cells == 0:
        return {"ergp_1": 0.0, "ergp_2": 0.0, "ergp_3": 0.0}

    results = {}
    for level in ["ergp_1", "ergp_2", "ergp_3"]:
        if level in erpg_thresholds:
            threshold = erpg_thresholds[level]
            exceedance = np.sum(concentration >= threshold)
            results[level] = float(exceedance) / float(total_cells)
        else:
            results[level] = 0.0

    return results