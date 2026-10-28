import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.basemap import Basemap


def _extract_band(ds, var, wavelength):
    """Nearest-wavelength 2D (lat, lon) slice as a raw numpy array."""
    return ds[var].sel(wavelength=wavelength, method="nearest").values


def _percentile_stretch(channel, valid_mask, percentile_low, percentile_high):
    """Linearly rescale one channel to [0, 1] using percentiles of valid pixels."""
    valid_vals = channel[valid_mask]
    lo, hi = np.nanpercentile(valid_vals, [percentile_low, percentile_high])
    return np.clip((channel - lo) / (hi - lo + 1e-12), 0, 1)


def _apply_gamma(channel, gamma):
    """Power-law brightness correction; NaNs pass through untouched."""
    return np.where(np.isfinite(channel), np.power(channel, gamma), np.nan)


def create_rgb(ds, var="Rrs",
               percentile_low=0.5,
               percentile_high=99.5,
               gamma=0.6,
               green_gamma_boost=0.7,
               blue_gamma_boost=0.95,
               ):
    """
    True-color RGB composite from a PACE-style xarray Dataset.
    Bands: 470 nm (B), 550 nm (G), 660 nm (R)

    Green and blue channels get an extra (lower) gamma on top of the base
    `gamma`, so subtle green/blue features (e.g. coccolithophore or
    chlorophyll signal) stay visible while red/yellow stay comparatively
    muted — without touching the percentile stretch or the function's
    inputs/outputs.
    """
    # ── 1. Extract each band as a 2D numpy array ───────────────────────────
    r = _extract_band(ds, var, 660.0)   # Red   channel
    g = _extract_band(ds, var, 550.0)   # Green channel
    b = _extract_band(ds, var, 470.0)   # Blue  channel

    # ── 2. Stack into (lat, lon, 3) — the shape imshow expects ─────────────
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

    # ── 3. Mask NaN → exclude from percentile calculation ──────────────────
    valid_mask = np.isfinite(rgb).all(axis=-1)

    # ── 4. Percentile contrast stretch (per-channel) ────────────────────────
    for i in range(3):
        rgb[:, :, i] = _percentile_stretch(
            rgb[:, :, i], valid_mask, percentile_low, percentile_high
        )

    # ── 5. Gamma correction, boosted for green/blue ─────────────────────────
    # Red keeps the base gamma; green and blue use a lower effective gamma
    # (base * boost factor) so they brighten more, making green/blue signal
    # stand out while red/yellow recede into the background.
    rgb[:, :, 0] = _apply_gamma(rgb[:, :, 0], gamma)                       # R
    rgb[:, :, 1] = _apply_gamma(rgb[:, :, 1], gamma * green_gamma_boost)   # G
    rgb[:, :, 2] = _apply_gamma(rgb[:, :, 2], gamma * blue_gamma_boost)    # B

    # ── 6. (Optional) NaN → white fill — left disabled as in the original ──
    # nan_pixels = ~np.isfinite(rgb).all(axis=-1)
    # rgb[nan_pixels] = 1.0

    # ── 7. Final clip to [0, 1] — power() can occasionally exceed 1 ────────
    rgb = np.clip(rgb, 0, 1)

    return rgb


def plot_shiptraject(lat, lon):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(lon, lat, marker='o', color='red', markersize=5)
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title="PACE RGB composite with ship trajectory")
    plt.tight_layout()
    return fig, ax

def rgb_plot_with_stations(
    ds,
    df,
    path=None,
    show_station_labels=True,
    show_stations=True,
    gridlines=True,
    save=False,
    lon_sat=None,
    lat_sat=None,
    extent=None,
    col_name_lat="lat",
    col_name_lon="lon",
    save_dir="/Users/timfugl/Desktop/PhD_Ocean_Color_Chemistry/Obj1_PIC Product Validation for PACE/Figures/Ocean Optics",
    title="PACE L2 RGB composite merged from 20260602 - 20260606",
    dpi = 1000,
):
    """
    Plots a PACE RGB composite with ship stations overlaid.

    Parameters:
    - ds: xarray Dataset containing the PACE data.
    - df: DataFrame containing station information with 'lat', 'lon', and 'Station_name'.
    - path: Optional path to a NetCDF/HDF file. If provided, ds is loaded from this path.
    - show_station_labels: Boolean to show station labels.
    - show_stations: Boolean to show station markers.
    - gridlines: Boolean to show gridlines.
    - save: Boolean to save the figure.
    - lon_sat: 1D or 2D array of satellite longitudes. If None, derived from ds.longitude.
    - lat_sat: 1D or 2D array of satellite latitudes. If None, derived from ds.latitude.
    - extent: [lon_min, lon_max, lat_min, lat_max]. If None, derived from lon_sat/lat_sat.
    - save_dir: Directory to save the figure.
    - title: Title string for the plot (also used as filename if save=True).

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import xarray as xr

    lat = df[col_name_lat].values
    lon = df[col_name_lon].values

    df["Station_ID"] = df["Station"].str[-2:]
    station_names = df["Station_ID"].astype(str)

    # Load dataset if path is provided
    if path is not None:
        ds = xr.open_dataset(path)

    # Derive satellite lat/lon if not provided
    if lon_sat is None:
        lon_sat = ds.longitude.values
    if lat_sat is None:
        lat_sat = ds.latitude.values

    # Derive extent if not provided
    if extent is None:
        extent = [lon_sat.min(), lon_sat.max(), lat_sat.min(), lat_sat.max()]

    # Prepare station data
    lat = df["lat"].values
    lon = df["lon"].values
    df["Station_ID"] = df["Station"].str[-2:]
    station_names = df["Station_ID"].astype(str)

    print(station_names.dtype)

    figsize = (10, 8)
    rgb = create_rgb(ds, var="Rrs",
             percentile_low=0.5, 
             percentile_high=99.5,
             gamma=0.8,
             )

    lon_1d = lon_sat
    lat_1d = lat_sat

    ny, nx, _ = rgb.shape

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Basemap with Mercator projection
    m = Basemap(
        projection="merc",
        llcrnrlat=extent[2],  # bottom (lat_min)
        urcrnrlat=extent[3],  # top    (lat_max)
        llcrnrlon=extent[0],  # left   (lon_min)
        urcrnrlon=extent[1],  # right  (lon_max)
        ax=ax,
        resolution="f",
    )

    # Coastlines etc.
    m.drawcoastlines(zorder=10, linewidth=0.5)
    m.drawcountries(zorder=10, linewidth=0.1)
    m.drawmapboundary(fill_color="none", zorder=10)
    m.fillcontinents(color="grey", alpha=0.3)

    parallels = np.arange(np.ceil(extent[2]), np.floor(extent[3]) + 2, 2)  # e.g. every 2 deg lat
    meridians = np.arange(np.ceil(extent[0]), np.floor(extent[1]) + 2, 2)  # e.g. every 2 deg lon

    if gridlines:
        # Draw latitude lines (parallels)
        m.drawparallels(
            parallels,
            labels=[True, False, False, False],  # label left
            linewidth=0.5,
            color="w",
            zorder=20,
        )

        # Draw longitude lines (meridians)
        m.drawmeridians(
            meridians,
            labels=[False, False, False, True],  # label bottom
            linewidth=0.5,
            color="w",
            zorder=20,
        )

    # Transform each RGB channel into the map projection
    rgb_proj = np.empty((ny, nx, 3), dtype=rgb.dtype)
    for i in range(3):
        rgb_proj[..., i] = m.transform_scalar(
            rgb[..., i],
            lon_1d,
            lat_1d,
            nx,
            ny,
        )

    # Now imshow in projected coordinates (no extent!)
    im = m.imshow(
        rgb_proj,
        origin="lower",
        interpolation="none",
        zorder=5,
    )

    # Stations: transform lon/lat → projected x/y
    if show_stations:
        x, y = m(lon, lat)

        ax.scatter(
            x,
            y,
            marker="x",
            color="red",
            s=70,
            zorder=20,
        )

    data_to_display = ax.transData
    display_to_data = ax.transData.inverted()

    if show_station_labels:
        for i, name in enumerate(station_names):
            # Projected map coordinate -> display coordinate
            x_display, y_display = data_to_display.transform(
                (x[i], y[i])
            )

            # Offset in display pixels
            x_display += 13
            y_display += -5

            # Display coordinate -> projected map coordinate
            x_text, y_text = display_to_data.transform(
                (x_display, y_display)
            )

            ax.text(
                x_text,
                y_text,
                name,
                ha="left",
                va="center",
                fontsize=11,
                fontstyle="oblique",
                color="black",
                zorder=21,
            )

    ax.set(
        title=title,
    )

    if save:
        plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    plt.show()