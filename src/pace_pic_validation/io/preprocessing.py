
import xarray as xr
import plotly
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from geopy.distance import geodesic
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import re
import glob
import imageio.v3 as iio
import subprocess
from joblib import delayed, Parallel
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objects as go
from pace_pic_validation.viz.visualization import plot_pic_world
import xarray as xr
import pandas as pd
import glob
import re
import re
from datetime import datetime
from pathlib import Path


from pathlib import Path


def open_l2(fpath, keep_vars=['Rrs', 'l2_flags'], target_wls=None):
    """ 
    Opens a PACE L2 file as an xarray dataset, assigning lat/lons (and wavelength, 
        if the dataset is 3D) as coordinates.
    Args:
        fpath - path to a L2 PACE file 
    Returns:
        ds - xarray dataset
    """
    # Open the file as a datatree to access the different groups
    # avoids using xarray's open_dataset(path, group = 'navigation_data') which doesn't handle the group structure of the file well
    # if opened with open_dataset, every group has to be handled seperately - there is no connection between groups, so lat/lons are not coordinates but just variables in the navigation_data group, and wavelength is not a coordinate but just a variable in the sensor_band_parameters group
    import xarray as xr
    
    dt = xr.open_datatree(fpath, engine="h5netcdf", decode_timedelta=False)

    #print(f"Opened datatree with groups: {list(dt.groups)}")
    #print(f'this is the raw datatree: {dt}')
    
    try:

        #print(f'before creating geo_ds from dt["geophysical_data"]: {dt["geophysical_data"]}')

        geo_ds = dt["geophysical_data"].to_dataset()

        # Keep only Rrs (and optionally l2_flags)
        vars_to_keep = keep_vars  # or ["Rrs", "l2_flags"]
        geo_ds = geo_ds[[v for v in vars_to_keep if v in geo_ds.data_vars]]

        print(f'geo_ds after creating from dt["geophysical_data"]: {geo_ds.vars}')


        if target_wls is not None:
            wls_coord = dt["sensor_band_parameters"].ds["wavelength_3d"]

            # wls_coord: (wavelength_3d) -> wavelength values

            # Assign wavelength values as coordinate on the wavelength_3d dimension
            geo_ds = geo_ds.assign_coords(wavelength_3d=("wavelength_3d", wls_coord.values))

            # Select along the spectral dimension
            geo_ds = geo_ds.sel(wavelength_3d=target_wls, method="nearest")

            print(f"Selected wavelengths: {geo_ds.wavelength_3d.values}")

            geo_ds = geo_ds.drop_vars("wavelength_3d")  # drop the original wavelength_3d variable if not needed

            sensor_coords = (
                dt["sensor_band_parameters"]
                .to_dataset()
                .sel(wavelength_3d=target_wls, method="nearest")
                .coords
            )

            sensor_coords = sensor_coords.drop_vars("wavelength")  # drop the original wavelength variable if not needed

            sensor_coords = sensor_coords[[v for v in vars_to_keep if v in sensor_coords.data_vars]]

            print(f"Sensor coordinates after selecting wavelengths: {sensor_coords}")


                    
        else:
            sensor_coords = dt["sensor_band_parameters"].to_dataset().coords

        ds = xr.merge((
            #dt.ds,  # this is the root group, with metadata
            dt.ds,

            #pulls actuall variables, but still without generic dimenstions: still number_of_lines, pixels_per_line, etc.
            geo_ds,

            #extracts wavelength axis etc and sets it as coordinates, but still with the original dimensions
            sensor_coords,  # safer than .coords directly

            #navigation data has lat/lon per number_of_lines and pixels_per_line, so we need to connect them as latitude and longitude coordinates to the geophysical data
            dt["navigation_data"].ds.set_coords(("longitude", "latitude")).coords,
        ))

    except:
        #this handles data which does not have the sensor_band_parameters group, so no wavelength coordinate, but is still useful for testing. this is for MODIS L2 data, which has the same structure but no sensor_band_parameters group
        ds = xr.merge((
            dt.ds,
            dt["geophysical_data"].to_dataset(),
            dt["navigation_data"].ds.set_coords(("longitude", "latitude")).coords,
        ))

    return ds


def open_l2_minimal(fpath, keep_vars=["Rrs", "l2_flags"], target_wls=None):
   
    dt = xr.open_datatree(fpath, engine="h5netcdf", decode_timedelta=False)

    # Geophysical variables
    geo_ds = dt["geophysical_data"].to_dataset()
    geo_ds = geo_ds[[v for v in keep_vars if v in geo_ds.data_vars]]

    # Navigation
    nav_coords = (
        dt["navigation_data"]
        .ds.set_coords(("longitude", "latitude"))
        .coords
    )

    # Wavelength coordinate (only what you need)
    wls = dt["sensor_band_parameters"].ds["wavelength_3d"]

    if target_wls is not None:

        # Select wavelengths (on the dimension, not on a coordinate)
        geo_ds = geo_ds.sel(wavelength=target_wls, method="nearest")



    # Merge only what you need
    ds = xr.merge((
        geo_ds,
        nav_coords,
    ), compat="override")

    return ds


def merge_simple(folder, variable_name, start_index=None, end_index=None):


    

    filenames = os.listdir(folder)
    filepaths = []
    print(filepaths)
    for filename in filenames: 
        if filename.startswith('.'):
            continue  # skip hidden files
        path = os.path.join(folder, filename)
        print(path)
        filepaths.append(path)

    filepaths = filepaths[start_index:end_index]  # limit to first 10 files for testing

    print(len(filepaths))


# 2) Open all files and stack them along a new dimension (e.g. "file")
    ds = xr.open_mfdataset(
        filepaths,
        combine="nested",      # important when there's no natural coord like time
        concat_dim="file",     # new dimension name
        data_vars="all",       # or ["your_var"] to be explicit
        coords="all",
        compat="override",     # skip strict checks since lat/lon are identical
    )

# 3) Take the mean over the file dimension for your variable
    var_mean = ds[variable_name].mean(dim="file")


# 4) making a xarray dataset from var_mean
    var_mean_ds = var_mean.to_dataset(name=variable_name)

    print(var_mean_ds)

    return var_mean_ds


def extract_start_date_from_nc(path: str) -> str | None:
    """
    Extract the start date from a filename like:
    2025-06-02_2025-06-06_latmin_48.0_..._['CLDICE'].nc
    Returns date as 'YYYYMMDD' or None if not found.
    """
    fname = Path(path).name
    m = re.match(r"(\d{4}-\d{2}-\d{2})", fname)
    if not m:
        return None
    date_str = m.group(1)  # "2025-06-02"
    # Convert to YYYYMMDD
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y%m%d")