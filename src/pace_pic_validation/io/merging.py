
from joblib import Parallel, delayed
import re
import numpy as np
import pandas as pd
import re
from typing import Union, List, Dict, Any
from pathlib import Path
import cartopy
import cartopy.crs as ccrs
import cf_xarray  # noqa: F401
import earthaccess
import matplotlib.pyplot as plt
import rasterio
import rioxarray as rio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.crs import CRS
import netCDF4
import h5netcdf
import os


# minimal function to reduce filesize of PACE L2. just extracts whats is in the list and target wls from RRS.
def decode_l2_flags(
    flags: Union[int, np.integer, List[int], np.ndarray],
) -> Union[List[str], List[List[str]]]:
    """
    Decode OBPG Level-2 integer flags into lists of flag abbreviations.

    - Single scalar (int or np.integer) -> list of flag names.
    - Array-like of ints -> list of lists of flag names.
    """
    OBPG_L2_FLAGS = {
    0:  "ATMFAIL",
    1:  "LAND",
    2:  "PRODWARN",
    3:  "HIGLINT",
    4:  "HILT",
    5:  "HISATZEN",
    6:  "COASTZ",
    # 7: spare
    8:  "STRAYLIGHT",
    9:  "CLDICE",
    10: "COCCOLITH",
    11: "TURBIDW",
    12: "HISOLZEN",
    # 13: spare
    14: "LOWLW",
    15: "CHLFAIL",
    16: "NAVWARN",
    17: "ABSAER",
    # 18: spare
    19: "MAXAERITER",
    20: "MODGLINT",
    21: "CHLWARN",
    22: "ATMWARN",
    # 23: spare
    24: "SEAICE",
    25: "NAVFAIL",
    26: "FILTER",
    # 27: spare
    28: "BOWTIEDEL",
    29: "HIPOL",
    30: "PRODFAIL",
    # 31: spare
    }
    def _decode_one(x: int) -> List[str]:
        x = int(x)  # ensure Python int
        return [name for bit, name in OBPG_L2_FLAGS.items() if (x >> bit) & 1]

    # Scalar case (Python int or NumPy scalar)
    if isinstance(flags, (int, np.integer)):
        return _decode_one(flags)

    # Array-like case
    return [_decode_one(x) for x in flags]

    """
    Decode OBPG Level-2 integer flags into lists of flag abbreviations.

    - Single scalar (int or np.integer) -> list of flag names.
    - Array-like of ints -> list of lists of flag names.
    """
    def _decode_one(x: int) -> List[str]:
        x = int(x)  # ensure Python int
        return [name for bit, name in OBPG_L2_FLAGS.items() if (x >> bit) & 1]

    # Scalar case (Python int or NumPy scalar)
    if isinstance(flags, (int, np.integer)):
        return _decode_one(flags)

    # Array-like case
    return [_decode_one(x) for x in flags]

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

def grid_data(src, resolution, dst_crs="epsg:4326", resampling=Resampling.nearest):
    
    #print(f'this is lat_min before gridding: {src["latitude"].min().values}')
    #rint(f'this is lat_max before gridding: {src["latitude"].max().values}')
    #print(f'this is lon_min before gridding: {src["longitude"].min().values}')
    #print(f'this is lon_max before gridding: {src["longitude"].max().values}')

    if (len(list(src.dims)) == 3) and (list(src.dims)[0] != "wavelength"):
        src = src.transpose("wavelength", ...)

    lon_2d = np.array(src["longitude"])
    lat_2d = np.array(src["latitude"])
    n_y, n_x = lon_2d.shape

    src = src.drop_vars(["longitude", "latitude"])
    src = src.rename_dims({"pixels_per_line": "x", "number_of_lines": "y"})
    src = src.assign_coords(x=np.arange(n_x), y=np.arange(n_y))
    src = src.rio.write_crs("epsg:4326")

    # KEY: pass src_geoloc_array directly — skip left/bottom/right/top
    defaults = rasterio.warp.calculate_default_transform(
        "epsg:4326",
        dst_crs,
        n_x,
        n_y,
        src_geoloc_array=(lon_2d, lat_2d),  # ← let rasterio compute bounds itself
    )
    #print("defaults width:", defaults[1], "height:", defaults[2])  # Verify

    transform, width, height = rasterio.warp.aligned_target(*defaults, resolution)
    #print("aligned width:", width, "height:", height)  # Should be ~3200, ~1600

    lon_da = xr.DataArray(lon_2d, dims=["y", "x"])
    lat_da = xr.DataArray(lat_2d, dims=["y", "x"])

    dst = src.rio.reproject(
        dst_crs=dst_crs,
        shape=(height, width),
        transform=transform,
        src_geoloc_array=(lon_da, lat_da),
        nodata=np.nan,
        resample=resampling,
    )
    dst["x"] = dst["x"].round(9)
    dst["y"] = dst["y"].round(9)

    dst = dst.rename({"x": "longitude", "y": "latitude"})

    #check if regridding worked.
    
    #print(f'this is lat_min after gridding: {dst["latitude"].min().values}')
    #print(f'this is lat_max after gridding: {dst["latitude"].max().values}')
    #print(f'this is lon_min after gridding: {dst["longitude"].min().values}')
    #print(f'this is lon_max after gridding: {dst["longitude"].max().values}')

    return dst


OBPG_L2_FLAGS: Dict[int, str] = {
    0:  "ATMFAIL",
    1:  "LAND",
    2:  "PRODWARN",
    3:  "HIGLINT",
    4:  "HILT",
    5:  "HISATZEN",
    6:  "COASTZ",
    # 7: spare
    8:  "STRAYLIGHT",
    9:  "CLDICE",
    10: "COCCOLITH",
    11: "TURBIDW",
    12: "HISOLZEN",
    # 13: spare
    14: "LOWLW",
    15: "CHLFAIL",
    16: "NAVWARN",
    17: "ABSAER",
    # 18: spare
    19: "MAXAERITER",
    20: "MODGLINT",
    21: "CHLWARN",
    22: "ATMWARN",
    # 23: spare
    24: "SEAICE",
    25: "NAVFAIL",
    26: "FILTER",
    # 27: spare
    28: "BOWTIEDEL",
    29: "HIPOL",
    30: "PRODFAIL",
    # 31: spare
}

# Reverse lookup: name -> bit
_FLAG_NAME_TO_BIT = {name: bit for bit, name in OBPG_L2_FLAGS.items()}


def _flags_to_mask(flags: Union[str, List[str]]) -> int:
    """
    Convert a single flag name or list of flag names to an integer mask.
    """
    if isinstance(flags, str):
        flags = [flags]

    mask = 0
    for name in flags:
        if name not in _FLAG_NAME_TO_BIT:
            raise ValueError(f"Unknown flag name: {name}")
        bit = _FLAG_NAME_TO_BIT[name]
        mask |= (1 << bit)
    return mask


def filter_by_l2_flags(
    ds: xr.Dataset,
    flags: Union[str, List[str]],
    reverse: bool = False,
) -> xr.Dataset:
    """
    Filter a dataset by OBPG L2 flags.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with an integer flag variable named "l2_flags".
    flags : str or list of str
        Flag name(s), e.g. "LAND" or ["LAND", "COASTZ"].
    reverse : bool, default False
        - False: keep pixels where NONE of the specified flags are set.
        - True:  keep pixels where ANY of the specified flags are set.

    Returns
    -------
    xr.Dataset
        Filtered dataset (with NaNs where the condition is not met).
    """
    if ds["l2_flags"].cf.is_flag_variable:
        # Build integer mask for the requested flag(s)
        mask = _flags_to_mask(flags)

        flags_var = ds["l2_flags"]

        if reverse:
            # Keep where ANY of the specified bits are set
            cond = (flags_var & mask) != 0
        else:
            # Keep where NONE of the specified bits are set
            cond = (flags_var & mask) == 0

        return ds.where(cond)

    else:
        print("l2_flags not recognized as flag variable")
        return ds

# ── Single-granule worker (must be top-level for joblib/loky pickling) ────────
def process_single_granule(path, lat_min, lat_max, lon_min, lon_max,
                            resolution, wavelengths, flag, reverse):
    
    import cf_xarray  # ← add this
    import numpy as np
    import pandas as pd
    import xarray as xr
    import re

    try:
        ds = open_l2_minimal(path, target_wls=wavelengths)

        line_mask = (
            (ds.latitude >= lat_min) & (ds.latitude <= lat_max) &
            (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
            
        ).any(dim="pixels_per_line")

        valid_lines = np.where(line_mask.values)[0]
        if len(valid_lines) == 0:
            
            #print(f"No overlap, skipping {path}")
            return None

        line_min_idx, line_max_idx = valid_lines[0], valid_lines[-1]
        ds = ds.isel(number_of_lines=slice(line_min_idx, line_max_idx + 1))
        print(f"Sliced to {line_max_idx - line_min_idx + 1} lines — {path}")

        ds_masked  = filter_by_l2_flags(ds, flags=flag, reverse=reverse)
        ds_masked  = ds_masked.transpose('wavelength', "number_of_lines", "pixels_per_line")
        ds_gridded = grid_data(ds_masked, resolution=resolution)

        ds_cropped = ds_gridded.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )

        match = re.search(r'\d{8}T\d{6}', path)
        timestamp = pd.to_datetime(match[0])

        # Attach timestamp as a scalar coordinate for tracking after merge
        ds_cropped = ds_cropped.assign_coords(granule_time=timestamp)

        return ds_cropped

    except Exception as e:
        print(f"Failed on {path}: {e}")
        return None

# ── Main merge function ───────────────────────────────────────────────────────
def granule_merge(directory, 
                  date_min, 
                  date_max, 
                  lat_min, 
                  lat_max, 
                  lon_min, 
                  lon_max,
                  resolution=(0.015, 0.015), 
                  wavelengths=None,
                  flags=["CLDICE"], 
                  reverse=False, 
                  n_jobs=2,
                  save_directory=None,
                ):

    # ── 1. Collect valid paths ────────────────────────────────────────────
    paths = []

    for filename in os.listdir(directory):
        if not filename.endswith('.nc') or filename.startswith('._'):
            continue
        match = re.search(r'\d{8}T\d{6}', filename)
        if not match:
            continue
        timestamp = pd.to_datetime(match[0])
        if pd.to_datetime(date_min) <= timestamp <= pd.to_datetime(date_max):
            full_path =os.path.join(directory, filename)
            paths.append(full_path)

    print(f"Found {len(paths)} granules in date range")

    if not paths:
        print("No granules found.")
        return None

    # ── 2. Process granules in parallel ───────────────────────────────────
    # loky = default joblib backend, spawns fresh processes (safe for xarray/numpy)
    # n_jobs=4 keeps memory ~10GB on a 16GB machine
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_single_granule)(
            path, lat_min, lat_max, lon_min, lon_max,
            resolution, wavelengths, flags, reverse
        )
        for path in paths
    )

    gridded_granules = [r for r in results if r is not None]

    # ── 3. Merge ──────────────────────────────────────────────────────────
    if not gridded_granules:
        print("No valid granules to merge.")
        return None
    
    batch_size = 2  # Adjust based on memory constraints
    print(f"Merging {len(gridded_granules)} granules in batches of {batch_size}")

    #creates batches of 20  
    path_batches = [
        gridded_granules[i:i+batch_size]
        for i in range(0, len(gridded_granules), batch_size)
    ]

    batch_means = []

    for b_idx, batch in enumerate(path_batches):
        print(f"Batch {b_idx+1}/{len(path_batches)} — {len(batch)} granules")

        batch_stack = xr.concat(batch, dim="granule_time")
        batch_mean  = batch_stack.mean(dim="granule_time", skipna=True)
        batch_means.append(batch_mean)

        #deletes memory
        
        del batch, batch_stack
        import gc; gc.collect()

    # Final merge of batch means (~18 tiny arrays)
    final = xr.concat(batch_means, dim="batch").mean(dim="batch", skipna=True)

    final.to_netcdf(f'{save_directory}/{date_min}_{date_max}_latmin_{lat_min:2.1f}_latmax_{lat_max:2.1f}_lonmin_{lon_min:2.1f}_lonmax_{lon_max:2.1f}_{str(flags)}.nc')

    return final


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