import json
import urllib.request
import numpy as np
import plotly.graph_objects as go
from typing import Sequence, Optional, Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import xarray as xr
from sklearn.neighbors import BallTree, KDTree
import pandas as pd



def _get_coastline_trace():

    path = '/Users/timfugl/Desktop/PhD_Ocean_Color_Chemistry/Obj1_PIC Product Validation for PACE/Data/Geo data/ne_10m_coastline.geojson'

    coast = json.load(open(path)
              )
    xs, ys = [], []
    print('opened url')
    print('start processing features')

    for feature in coast["features"]:
        geom = feature["geometry"]

        if geom["type"] == "LineString":
            coords = geom["coordinates"]
            for x, y in coords:
                xs.append(x)
                ys.append(y)
            xs.append(None)
            ys.append(None)

        elif geom["type"] == "MultiLineString":
            for line in geom["coordinates"]:
                for x, y in line:
                    xs.append(x)
                    ys.append(y)
                xs.append(None)
                ys.append(None)

    print('finished processing features')
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color="black", width=1),
        hoverinfo="skip",
        showlegend=False,
        name="Coastline",
    )


def plot_pic_world(ds, var="PIC_mean", step=8,
                   lat_min=-90, lat_max=90,
                   lon_min=-180, lon_max=180,
                   title=None):

    sub = ds.sel(
        lat=slice(lat_max, lat_min),
        lon=slice(lon_min, lon_max)
    ).isel(
        lat=slice(None, None, step),
        lon=slice(None, None, step)
    )


    z = sub[var].values * 1000
    lon = sub["lon"].values
    lat = sub["lat"].values


    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=lon,
            y=lat,
            z=z,
            colorscale="Viridis",
            zmin=0,
            zmax=2,
            colorbar=dict(title=var),
            hovertemplate=(
                "lon: %{x:.3f}<br>"
                "lat: %{y:.3f}<br>"
                "PIC: %{z:.4g}<extra></extra>"
            ),
        )
    )


    
    fig.add_trace(_get_coastline_trace())

    fig.update_layout(
        title=title or var,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        width=1000,
        height=700,
    )

    fig.update_xaxes(range=[lon_min, lon_max])
    fig.update_yaxes(
        range=[lat_min, lat_max],
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def plot_pic_with_stations(ds_sat, ds_stations, lat_sat_var = 'lat', lon_sat_var = 'lon', pic_sat_var = 'pic', lat_station_var = 'Lat', lon_station_var = 'Lon', station_name_var = 'station',   title = 'PIC MODIS 26.7–8.8'): 

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

# assign vars
    sat = ds_sat
    lat = sat[lat_sat_var].values        # 1D lat in degrees
    lon = sat[lon_sat_var].values        # 1D lon in degrees
    PIC_8D = sat[pic_sat_var].values * 1000     # 2D field [lat, lon]

    lat_stat = ds_stations[lat_station_var].values   # station lat in degrees
    lon_stat = ds_stations[lon_station_var].values   # station lon in degrees

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={'projection': ccrs.Stereographic()}
    )

    # base map
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0, alpha=0.2)
    ax.add_feature(cfeature.LAND, facecolor='#90FF15', zorder=1, alpha=0.2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, zorder=2)

    # extent is given in geodetic (lon/lat) coordinates
    ax.set_extent([0, 25, 65, 80], crs=ccrs.PlateCarree())

    # station points (x=lon, y=lat; data CRS = PlateCarree)
    ax.scatter(
        lon_stat, lat_stat,
        color='red', s=10,
        transform=ccrs.PlateCarree(),
        zorder=5, label='Stations'
    )
    ax.plot(
        lon_stat, lat_stat,
        color='red',
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # station labels
    for i, name in enumerate(ds_stations[station_name_var]):
        ax.text(
            lon_stat[i] - 0.1, lat_stat[i] - 0.2,  # small lat offset
            name,
            fontsize=8,
            fontstyle='oblique',
            color='red',
            transform=ccrs.PlateCarree(),
            zorder=6
        )

    # Create 2D coordinate grids from lon/lat
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # pcolormesh: data are lon/lat in degrees => PlateCarree transform
    im = ax.pcolormesh(
        lon_2d, lat_2d, PIC_8D,
        transform=ccrs.PlateCarree(),
        zorder=3, vmax=2, cmap='viridis'
    )

    plt.colorbar(im, ax=ax, label='PIC')
    ax.set_title(title)

    plt.show()

def xy_dalec(dalec, lat, lon):

    #find closest row

    #converting all lats and lons to radians

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    for column in ['LATITUDE', 'LONGITUDE']:
        rad = np.deg2rad(dalec[column].values)
        dalec[f'{column}_rad'] = rad

    ball = BallTree(dalec[["LATITUDE_rad", "LONGITUDE_rad"]].values, metric='haversine')

    distance, index = ball.query([[lat_rad, lon_rad]], k=1)

    distance_km = distance[0, 0] * 6371.0088
    index = index[0,0]

    datetime = dalec.index[index]

    #print(f'DALEC datetime: {datetime}')
    #print(f'DALEC coord: {dalec['LATITUDE'][index], dalec['LONGITUDE'][index]} ]')
    #print(f'Station coord: {lat, lon} ]')
    #print(f'distance from Station: {distance_km}')

    # Convert index once
    dalec = dalec.copy()

    dalec.index = pd.to_datetime(dalec.index, utc=True)

    print(dalec.index)

    dalec = dalec.drop(columns = ['LATITUDE', 'LONGITUDE', 'LATITUDE_rad', 'LONGITUDE_rad'])
    row = dalec.loc[datetime]

    x = dalec.columns
    y = row

    return x,y 
        

def plot_rrs_spectra(
    ds: xr.Dataset,
    lons: Sequence[float],
    lats: Sequence[float],
    stations: Sequence[str],
    rrs_variable: str = "Rrs",
    wavelength_variable: str = "wavelength",
    mode: str = "multi",  # "single" or "multi"
    figsize: Tuple[float, float] = (6.5, 4.5),
    dpi: int = 150,
    xlim: Optional[Tuple[float, float]] = (400, 700),
    ylim: Optional[Tuple[float, float]] = None,
    xticks: Optional[Sequence[float]] = None,
    xlabel: str = "Wavelength (nm)",
    ylabel: str = r"Remote-sensing reflectance, $R_{rs}$ (sr$^{-1}$)",
    title: Optional[str] = None,
    legend_title: str = "Station",
    cmap: str = "viridis",
    cmap_range: Tuple[float, float] = (0, 0.9),
    line_kwargs: Optional[dict] = None,
    grid: bool = True,
    grid_kwargs: Optional[dict] = None,
    minor_ticks: bool = True,
    spines_top_right: bool = True,
    save_dir: Optional[str] = None,
    save_kwargs: Optional[dict] = None,
    show: bool = True,
    apply_style: bool = True,
    dalec = None,
) -> plt.Figure:
    """
    Plot Rrs spectra for one or multiple stations.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing the Rrs variable and wavelength coordinate.
    lons : sequence of float
        Longitudes for each station.
    lats : sequence of float
        Latitudes for each station.
    stations : sequence of str
        Station names/labels for each location.
    rrs_variable : str, optional
        Name of the Rrs variable in the Dataset.
    wavelength_variable : str, optional
        Name of the wavelength coordinate/variable.
    mode : {"single", "multi"}, optional
        "single" → create one figure per station (calls plt.show() each time).
        "multi" → plot all stations on a single figure.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    dpi : int, optional
        Figure resolution.
    xlim : tuple or None, optional
        (xmin, xmax) for x-axis. Set to None to auto-scale.
    ylim : tuple or None, optional
        (ymin, ymax) for y-axis. Set to None to auto-scale.
    xticks : sequence or None, optional
        Tick locations on x-axis.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    title : str or None, optional
        Figure title. If None, no title is added.
    legend_title : str, optional
        Title for the legend.
    cmap : str, optional
        Matplotlib colormap for station colors.
    cmap_range : tuple, optional
        Fractional range of colormap to use, e.g. (0, 0.9).
    line_kwargs : dict or None, optional
        Additional keyword arguments passed to `ax.plot()` for each spectrum.
        Example: {"linewidth": 0.6, "alpha": 0.85}.
    grid : bool, optional
        Whether to draw a grid.
    grid_kwargs : dict or None, optional
        Keyword arguments for `ax.grid()`, e.g. {"ls": ":", "alpha": 0.9}.
    minor_ticks : bool, optional
        Whether to add minor ticks on both axes.
    spines_top_right : bool, optional
        Whether to show top and right spines.
    save_path : str or None, optional
        Path (including filename) to save the figure. If None, figure is not saved.
    save_kwargs : dict or None, optional
        Additional keyword arguments for `fig.savefig()`, e.g. {"dpi": 300}.
    show : bool, optional
        Whether to call `plt.show()` at the end.
    apply_style : bool, optional
        Whether to apply a publication-style `rcParams` update.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """

    if len(lons) != len(lats) or len(lons) != len(stations):
        raise ValueError("lons, lats, and stations must have the same length.")

    if line_kwargs is None:
        line_kwargs = {}

    if grid_kwargs is None:
        grid_kwargs = {"ls": ":", "alpha": 0.9}

    if save_kwargs is None:
        save_kwargs = {}

    if apply_style:
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.size": 11,
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Arial",
                    "DejaVu Sans",
                    "Liberation Sans",
                ],
                "axes.linewidth": 1.2,
                "axes.labelsize": 12,
                "axes.titlelocation": "left",
                "axes.titleweight": "normal",
                "xtick.major.size": 6,
                "xtick.major.width": 1.2,
                "xtick.minor.size": 3,
                "xtick.minor.width": 1.0,
                "ytick.major.size": 6,
                "ytick.major.width": 1.2,
                "ytick.minor.size": 3,
                "ytick.minor.width": 1.0,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
                "legend.frameon": False,
                "legend.fontsize": 10,
            }
        )

   


    if mode == "single":
        # One figure per station

        for lon, lat, station in zip(lons, lats, stations):
            test = ds[rrs_variable].sel(
                longitude=lon,
                latitude=lat,
                method="nearest",
            )

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            ax.plot(
                test[wavelength_variable].values,
                test.values,
                **line_kwargs,
            )

            if dalec is not None:

                  # or ':', '-.', etc.
                x,y = xy_dalec(dalec, lat, lon)
                ax.plot(
                        x,
                        y,
                        color="black",           # or another fixed color
                        linestyle="--",          # explicit, in case line_kwargs doesn't set it
                        label="DALEC",           # if you want it in the legend
                    )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if xlim is not None:
                ax.set_xlim(xlim)
            if xticks is not None:
                ax.set_xticks(xticks)
            if ylim is not None:
                ax.set_ylim(ylim)

            if title is not None:
                ax.set_title(title)
            else:
                ax.set_title(f"Station: {station}")

            if grid:
                ax.grid(**grid_kwargs)

            if minor_ticks:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            if spines_top_right:
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)

            plt.tight_layout()

            if save_dir is not None:
                # For single mode, you might want to include station in filename
                fig.savefig(f'{save_dir}/{station}_rrs_spectrum.png', dpi=save_kwargs.get("dpi", dpi), bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close(fig)

        # Return last figure (or None if you prefer)
        return fig

    elif mode == "multi":
        # All spectra in one plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        colors = plt.cm.get_cmap(cmap)(np.linspace(cmap_range[0], cmap_range[1], len(stations)))

        for i, (lon, lat, station) in enumerate(zip(lons, lats, stations)):
            test = ds[rrs_variable].sel(
                longitude=lon,
                latitude=lat,
                method="nearest",
            )

            ax.plot(
                test[wavelength_variable].values,
                test.values,
                color=colors[i],
                label=f"{station}",
                **line_kwargs,
            )

        if dalec is not None:
                # Prepare DALEC-specific line style # or ':', '-.', etc.

            for lon, lat in zip(lons, lats):  # adapt to your DALEC loop

                x, y = xy_dalec(dalec, lat, lon)

                ax.plot(
                        x,
                        y,
                        linestyle="--",          # explicit, in case line_kwargs doesn't set it
                        label="DALEC",           # if you want it in the legend
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if ylim is not None:
            ax.set_ylim(ylim)

        if title is not None:
            ax.set_title(title)

        if grid:
            ax.grid(**grid_kwargs)

        if minor_ticks:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

        if spines_top_right:
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

        if len(stations) > 0:
            ax.legend(title=legend_title, loc="upper right", frameon=False)

        plt.tight_layout()

        if save_dir is not None:
            fig.savefig(f"{save_dir}{stations[0][0:10]}_rrs_spectra_all.png", dpi=save_kwargs.get("dpi", dpi), bbox_inches="tight")

        if show:
            plt.show()

        return fig

    else:
        raise ValueError("mode must be 'single' or 'multi'.")



def pace_granule_plot_prep(granule_path, lon, lat):

    from pace_pic_validation.io.preprocessing import open_l2_minimal
    from pace_pic_validation.matchup.matchup_function import decode_l2_flags
    import matplotlib.pyplot as plt 

    ds = open_l2_minimal(granule_path)
 
    dlat = ds.latitude - lat
    dlon = ds.longitude - lon

    dist = np.sqrt(dlat**2 + (dlon * np.cos(np.deg2rad(lat)))**2)
    dist_km = dist * (np.pi / 180.0) * 6378 

    idx = dist.argmin(dim=["number_of_lines", "pixels_per_line"])

    ds_pt = ds.isel(
    number_of_lines=idx["number_of_lines"],
    pixels_per_line=idx["pixels_per_line"],
    )
    flag_int = [ds_pt['l2_flags'].values]
    lat_pace = ds_pt['latitude'].values
    lon_pace = ds_pt['longitude'].values
    flags_list = decode_l2_flags(flag_int)

    print(f'{flags_list} for spectrum at {lat_pace}, {lon_pace}, distance to desired lat lon is {dist_km.min().item()} km')
    x = ds_pt['wavelength']
    y = ds_pt['Rrs']

    return x, y, flags_list

