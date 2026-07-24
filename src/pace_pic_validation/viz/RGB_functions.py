def plot_rgb(ds, var="Rrs",
             percentile_low=2, percentile_high=98,
             gamma=0.5,
             figsize=(14, 8)):
    """
    True-color RGB composite from a PACE-style xarray Dataset.
    Bands: 470 nm (B), 550 nm (G), 660 nm (R)
    """

    # ── 1. Extract each band as a 2D numpy array ──────────────────────────────
    r = ds[var].sel(wavelength_3d=660.0).values   # Red   channel
    g = ds[var].sel(wavelength_3d=550.0).values   # Green channel
    b = ds[var].sel(wavelength_3d=470.0).values   # Blue  channel
    # .sel() does a label-based lookup on the wavelength coord.
    # .values drops xarray overhead and gives a raw (lat, lon) numpy array.

    # ── 2. Stack into (lat, lon, 3) — the shape imshow expects ────────────────
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    # axis=-1 appends the new channel axis as the last dimension.
    # float32 saves memory (vs float64) and is enough for display.

    # ── 3. Mask NaN → 0 temporarily for percentile calculation ───────────────
    valid_mask = np.isfinite(rgb).all(axis=-1)   # True where all 3 bands are valid
    rgb_valid  = rgb[valid_mask]                 # Flatten to (N, 3) for np.percentile

    # ── 4. Percentile contrast stretch (per-channel) ──────────────────────────
    for i in range(3):
        lo, hi = np.nanpercentile(rgb_valid[:, i],
                                  [percentile_low, percentile_high])
        rgb[:, :, i] = np.clip((rgb[:, :, i] - lo) / (hi - lo + 1e-12),
                                0, 1)
    # np.nanpercentile ignores NaNs and returns the pLow and pHigh reflectance
    # thresholds. Subtracting lo and dividing by the range rescales to [0, 1].
    # np.clip hard-clips outliers outside that range.
    # 1e-12 avoids division-by-zero when a band is flat/missing.

    # ── 5. Gamma correction ───────────────────────────────────────────────────
    rgb = np.where(np.isfinite(rgb), np.power(rgb, gamma), np.nan)
    # Ocean Rrs values are tiny (~0–0.03 sr⁻¹), so most pixels cluster near 0
    # after stretching. gamma < 1 brightens mid-tones non-linearly:
    #   pixel_out = pixel_in ^ gamma
    # gamma=0.5 (square root) is a good starting point for ocean color.
    # gamma=1.0 → no correction; gamma=0.3 → very bright (useful for dark scenes).

    # ── 6. Replace remaining NaN with white (1,1,1) for clean rendering ───────
    nan_pixels = ~np.isfinite(rgb).all(axis=-1)
    rgb[nan_pixels] = 1.0   # White background. Use 0.0 for black (clouds stand out).

    # ── 7. Final clip to [0,1] — power() can occasionally exceed 1 ────────────
    rgb = np.clip(rgb, 0, 1)

    # ── 8. Build georeferenced axes and plot ──────────────────────────────────
    lon = ds.longitude.values
    lat = ds.latitude.values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb,
              extent=extent,
              origin="lower",      # latitude decreases downward in imshow by default
              aspect="auto",       # don't force square pixels (lat/lon are not equal-area)
              interpolation="none" # no blurring; keep raw pixel fidelity
              )
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title="PACE RGB composite — North Sea / NE Atlantic")
    plt.tight_layout()
    plt.show()
    return fig, ax