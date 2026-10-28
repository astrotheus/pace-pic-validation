# match up in-situ data with 3x3 satellite pixel for variability check
# output should give 3x3 pixel with flag mask + matched in-situ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist    
import xarray as xr 
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree
import glob
import re
import os
from collections import defaultdict
import netCDF4 as nc
import shutil
from typing import Union, List


def open_l2(fpath, target_wls=None, MODIS_CI2 = False):
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
    dt = xr.open_datatree(fpath, engine="h5netcdf", decode_timedelta=False)

    print(f"Opened datatree with groups: {list(dt.groups)}")
    #print(f'this is the raw datatree: {dt}')
    
    try:

        #print(f'before creating geo_ds from dt["geophysical_data"]: {dt["geophysical_data"]}')

        geo_ds = dt["geophysical_data"].to_dataset()

        #print(f'geo_ds after creating from dt["geophysical_data"]: {geo_ds}')


        if target_wls is not None:
            
            #assigning the coordinate for a wavelength 
            #this takes the 172 values (dimension of wavelength) of sensor band parameters which holds the nanometer values. wls_coords then is a 172 array with wavelength values.
            wls_coord = dt["sensor_band_parameters"].ds["wavelength"]
            #dt.ds is accessing the subgroup and displaying /converting content as a dataset. this is data structure.
            #the dataset is like a ttribute to the datatree.

            #this is the structure of the datatree:
            # datatree/
            #   ├── root (dt.ds)                    ← metadata (root group)
            #   ├── geophysical_data (dt[...].ds)   ← Rrs(..., wavelength: 172)
            #   ├── sensor_band_parameters (...)    ← wavelength=[346..719]
            #   └── navigation_data (...)           ← lat, lon

            #dimension is indexes: accesible with isel(wavelength=0) while sel(wavelength=470, method='nearest') searches for coordinate value.
            
            # Use sel(method='nearest') on coord FIRST
            sel_wls = wls_coord.sel(wavelength=target_wls, method='nearest')

            ##print(f"Selected wavelengths: {sel_wls.values}")

            geo_ds = geo_ds.assign_coords(wavelength=wls_coord)
            geo_ds = geo_ds.sel(wavelength=sel_wls, method='nearest')  # or isel

            #print(f"Geo dataset wavelengths: {geo_ds.wavelength.values}")
            
            sensor_coords = dt["sensor_band_parameters"].to_dataset().sel(wavelength=sel_wls).coords
            
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

    except Exception as e:
        #print(f"Falling back to MODIS structure: {e}")
        #this handles data which does not have the sensor_band_parameters group, so no wavelength coordinate, but is still useful for testing. this is for MODIS L2 data, which has the same structure but no sensor_band_parameters group
        ds = xr.merge((
            dt.ds,
            dt["geophysical_data"].to_dataset(),
            dt["navigation_data"].ds.set_coords(("longitude", "latitude")).coords,
        ))

        if MODIS_CI2:
            ds = ds[["Rrs_667", "Rrs_547"]]
        
    return ds

# OBPG Level-2 flag definitions (bit -> name)
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

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])


# ── Helper: load and validate granules ────────────────────────────────────
def _load_valid_granules(L2_granule_input: Union[str, List[str], ], quick_check_remote: bool = False,) -> List[str]:
    """
    Load valid granule paths/URLs.

    Accepts:
      - A local directory path (str): returns local file paths to valid .nc files.
      - A list of HTTPS URLs (List[str]): returns the subset of URLs that can be opened.

    Returns:
      List of valid file paths or URLs (strings).
    """
    valid_files = []

    # Case 1: list of URLs or paths
    if isinstance(L2_granule_input, list):
        if not quick_check_remote:
            # Just return all; avoid opening every remote file
            return L2_granule_input
        
        for obj in L2_granule_input:
            try:
                ds = xr.open_dataset(obj)
                # Minimal read to ensure it's usable
                ds.close()
                valid_files.append(obj)
                print(f'{obj} is valid.')
            except OSError as e:
                print(f"❌ Corrupted/unreadable file: {obj}")
                print(f"   Error: {e}")
        return valid_files

    # Case 2: local directory path
    elif isinstance(L2_granule_input, str):
        for fpath in sorted(glob.glob(os.path.join(L2_granule_input, '*.nc'))):
            try:
                with nc.Dataset(fpath, 'r') as ds:
                    valid_files.append(fpath)
            except OSError as e:
                print(f"❌ Corrupted file: {fpath}")
                print(f"   Error: {e}")
        return valid_files

    else:
        raise TypeError(
            "L2_granule_input must be a local directory path (str) "
            "or a list of file paths/URLs (List[str])."
        )


# ── Helper: extract satellite datetimes from filenames ───────────────────
def _extract_sat_datetimes(valid_files):
    datetimes_sat = []
    for fpath in valid_files:
        fpath = str(fpath)
        match = re.search(r'\d{8}T\d{6}', fpath)
        dt = pd.to_datetime(match.group(), format='%Y%m%dT%H%M%S')
        datetimes_sat.append(dt)
    datetimes_sat.sort()

    print(f'length of sat datetimes: {len(datetimes_sat)}')

    return datetimes_sat


# ── Helper: build time tree and basic arrays ─────────────────────────────
def _prepare_insitu_arrays(in_situ_timestamps, in_situ_lat, in_situ_lon, time_threshold_min):
    insitu_datetimes = pd.to_datetime(in_situ_timestamps, format='mixed')
    insitu_datetimes = np.array(insitu_datetimes.values)
    insitu_times_sec = insitu_datetimes.astype('int64') / 1e9  # ns → seconds
    #print(f'length of insitu datetimes: {len(insitu_times_sec)}')
    insitu_timetree = cKDTree(insitu_times_sec.reshape(-1, 1))
    insitu_lat = np.array(in_situ_lat)
    insitu_lon = np.array(in_situ_lon)
    time_delta_secs = time_threshold_min * 60
    return insitu_times_sec, insitu_timetree, insitu_lat, insitu_lon, time_delta_secs


# ── Helper: process one granule for time+spatial matching ────────────────
def _process_granule_for_matches(
    fpath,
    insitu_times_sec,
    insitu_timetree,
    insitu_lat,
    insitu_lon,
    time_delta_secs,
    spatial_threshold_km,
    best_matches,
    moving=False,
    sat_match_filepath=None,
    delete_no_match_sat_files=False
):
    
    string = str(fpath)
    match = re.search(r'\d{8}T\d{6}', string)
    print(match.group(0))
    if not match:
        return

    timestamp_sat = pd.to_datetime(match.group(), format='%Y%m%dT%H%M%S')
    timestamp_sat_sec = np.array([[timestamp_sat.value / 1e9]])
    indices_time = insitu_timetree.query_ball_point(timestamp_sat_sec, r=time_delta_secs)[0]

    if len(indices_time) == 0:
        print(f"No in-situ matches found for granule {fpath} within time threshold.")
        if delete_no_match_sat_files:
            os.remove(fpath)
            print(f'deleted {fpath}')
        return

    if moving:
        if len(indices_time) > 0:
            print(f"in-situ matches found for granule {fpath} within time threshold.")
            shutil.move(fpath, sat_match_filepath)
            print(f'sat file moved to {sat_match_filepath}')
        return

    nav = xr.open_dataset(fpath, group="navigation_data", engine="h5netcdf")
    lat_sat = nav["latitude"].values
    lon_sat = nav["longitude"].values
    swath_shape = lat_sat.shape
    nav.close()

    lat_flat = lat_sat.ravel()
    lon_flat = lon_sat.ravel()
    valid_geo = np.isfinite(lat_flat) & np.isfinite(lon_flat)
    valid_flat_indices = np.where(valid_geo)[0]
    spatial_tree = cKDTree(latlon_to_xyz(lat_flat[valid_geo], lon_flat[valid_geo]))

    xyz_insitu = latlon_to_xyz(insitu_lat[indices_time], insitu_lon[indices_time])
    dist_3d, pix_in_valid = spatial_tree.query(xyz_insitu, k=1)
    pix_flat = valid_flat_indices[pix_in_valid]

    R_earth = 6371.0
    max_chord = 2 * np.sin(spatial_threshold_km / (2 * R_earth))
    valid_mask = dist_3d < max_chord

    indices_time_valid = np.array(indices_time)[valid_mask]
    pix_flat_valid = pix_flat[valid_mask]

    if len(indices_time_valid) == 0:
        return

    for insitu_idx, pix in zip(indices_time_valid, pix_flat_valid):
        time_diff = abs(timestamp_sat_sec[0, 0] - insitu_times_sec[insitu_idx])

        if insitu_idx not in best_matches or time_diff < best_matches[insitu_idx]["time_diff"]:
            best_matches[insitu_idx] = {
                "time_diff": time_diff / 60,  # stored in minutes
                "pix_flat": int(pix),
                "fpath": fpath,
                "swath_shape": swath_shape
            }


# ── Helper: build granule groups from best_matches ───────────────────────
def _build_granule_groups(best_matches):
    granule_groups = defaultdict(list)
    for insitu_idx, m in best_matches.items():
        granule_groups[m["fpath"]].append(insitu_idx)
        
    insitu_indices = list(best_matches.keys())
    return granule_groups, insitu_indices


# ── Helper: build a single record for one insitu_idx & granule ───────────
def _build_record(
    insitu_idx,
    fpath,
    ds,
    best_matches,
    insitu_lat,
    insitu_lon,
    in_situ_value,
    flag_array,
    flag_mask,
    is_MODIS,
    target_wls,
    min_valid_pixels,
    in_situ_timestamps,
):
    pix_flat = best_matches[insitu_idx]["pix_flat"]
    swath_shape = best_matches[insitu_idx]["swath_shape"]
    nrows, ncols = swath_shape
    pix_i, pix_j = np.unravel_index(pix_flat, swath_shape)


    if pix_i < 1 or pix_i >= nrows - 1 or pix_j < 1 or pix_j >= ncols - 1:
        return None

    row_sl = slice(pix_i - 1, pix_i + 2)
    col_sl = slice(pix_j - 1, pix_j + 2)

    insitu_idx = int(insitu_idx)



    record = {
        "insitu_idx": insitu_idx,
        "lat_insitu": insitu_lat[insitu_idx],
        "lon_insitu": insitu_lon[insitu_idx],
        "insitu_value": float(in_situ_value[insitu_idx]),
        "time_diff_min": best_matches[insitu_idx]["time_diff"],
        "insitu_timestamp": in_situ_timestamps[insitu_idx],
        "granule": fpath,
    }

    # ── Flags: always collect raw integer flag per pixel (no masking) ──

    if flag_array is not None:
        flag_box = flag_array[row_sl, col_sl]          # shape (3, 3)
        for px_idx, px_val in enumerate(flag_box.ravel()):
            record[f"flag_px{px_idx}"] = int(px_val)
            flag_list = decode_l2_flags(px_val)
            record[f"flag_px_decode{px_idx}"] = flag_list
  

    else:
        for px_idx in range(9):
            record[f"flag_px{px_idx}"] = np.nan        # or -1 if you prefer a sentinel int
        print("no flag_array available; flags set to NaN")

    # ── Per-wavelength stats ───────────────────────────────────────
    skip_record = False

    wl_var_pairs = zip([547, 667], ["Rrs_547", "Rrs_667"]) if is_MODIS else \
                   ((wl, "Rrs") for wl in target_wls)

    for wl, var in wl_var_pairs:
        if is_MODIS:
            box = ds[var].values[row_sl, col_sl].astype(float).copy()

        else:

            box = (ds["Rrs"]
                   .sel(wavelength=wl, method="nearest")
                   .values[row_sl, col_sl].astype(float).copy())

        # ── Store raw unmasked values BEFORE applying flags ────────
        for px_idx, px_val in enumerate(box.ravel()):
            record[f"Rrs_{wl}_px{px_idx}_raw"] = float(px_val)

        valid_n = int(np.sum(~np.isnan(box)))

        mean = np.nanmean(box)
        std = np.nanstd(box)
        record[f"Rrs_{wl}_mean"] = mean
        record[f"Rrs_{wl}_std"] = std
        record[f"Rrs_{wl}_median"] = np.nanmedian(box)
        record[f"Rrs_{wl}_valid_n"] = valid_n
        record[f"Rrs_{wl}_cv"] = std / mean if (mean != 0 and not np.isnan(mean)) else np.nan



    record["sat_lat"] = float(ds["latitude"].values[pix_i, pix_j])
    record["sat_lon"] = float(ds["longitude"].values[pix_i, pix_j])

    # Haversine distance between matched satellite pixel and in-situ point

    lat1 = np.radians(insitu_lat[insitu_idx])
    lon1 = np.radians(insitu_lon[insitu_idx])
    lat2 = np.radians(record["sat_lat"])
    lon2 = np.radians(record["sat_lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    record["spatial_dist_km"] = float(2 * 6371.0 * np.arcsin(np.sqrt(a)))

    return record


# ── Helper: process one granule group into records ───────────────────────
def _process_granule_group(
    fpath,
    insitu_indices,
    best_matches,
    insitu_lat,
    insitu_lon,
    in_situ_value,
    is_MODIS,
    target_wls,
    flag_mask,
    min_valid_pixels,
    in_situ_timestamps
):
    #print(f'opening {fpath}')

    if is_MODIS:
        ds = open_l2(fpath, MODIS_CI2=True)
        #print('assumption: MODIS file')

    else:

        ds = open_l2(fpath, target_wls=target_wls)
        #print('assumption: PACE file')

  
    swath_shape = best_matches[insitu_indices[0]]["swath_shape"]
    nrows, ncols = swath_shape
  
    flag_array = ds["l2_flags"].values
    
    results = []

    for insitu_idx in insitu_indices:
        record = _build_record(
            insitu_idx=insitu_idx,
            fpath=fpath,
            ds=ds,
            best_matches=best_matches,
            insitu_lat=insitu_lat,
            insitu_lon=insitu_lon,
            in_situ_value=in_situ_value,
            flag_array=flag_array,
            flag_mask=flag_mask,
            is_MODIS=is_MODIS,
            target_wls=target_wls,
            min_valid_pixels=min_valid_pixels,
            in_situ_timestamps = in_situ_timestamps,
        )
        if record is not None:
            results.append(record)

    ds.close()
    return results


# ── Core matching function (unchanged logic, now calls helpers) ───────────
def matchup3x3(L2_granule_directory, 
               in_situ_timestamps, 
               in_situ_lat, 
               in_situ_lon, 
               in_situ_value,
               time_threshold_min, 
               spatial_threshold_km,
               target_wls=[547, 667], 
               is_MODIS=False,
               flag_mask=None, 
               min_valid_pixels=5, 
               delete_no_match_sat_files = False,
               sat_match_filepath = None,
               moving = False):

    # flag_mask: e.g. 0x0200 | 0x0008 | 0x0001  →  CLDICE | HIGLINT | LAND
    # min_valid_pixels: minimum unflagged pixels required per band (Bailey & Werdell: 5/9)
    # raw Rrs values (Rrs_{wl}_px{0-8}_raw) are always stored before flag masking
    # so any flag mask can be reapplied post-hoc from the output DataFrame

    print('variables assigned')

    insitu_times_sec, insitu_timetree, insitu_lat, insitu_lon, time_delta_secs = _prepare_insitu_arrays(
        in_situ_timestamps, in_situ_lat, in_situ_lon, time_threshold_min
    )

    print(f"Searching for matches in {L2_granule_directory} with time threshold {time_threshold_min} min and spatial threshold {spatial_threshold_km} km...")

    valid_files = _load_valid_granules(L2_granule_directory)

    datetimes_sat = _extract_sat_datetimes(valid_files)
    print(f'sat files range from {datetimes_sat[0]} to {datetimes_sat[-1]}')

    best_matches = {}

    for fpath in valid_files:
        _process_granule_for_matches(
            fpath=fpath,
            insitu_times_sec=insitu_times_sec,
            insitu_timetree=insitu_timetree,
            insitu_lat=insitu_lat,
            insitu_lon=insitu_lon,
            time_delta_secs=time_delta_secs,
            spatial_threshold_km=spatial_threshold_km,
            best_matches=best_matches,
            moving=moving,
            sat_match_filepath=sat_match_filepath,
            delete_no_match_sat_files=delete_no_match_sat_files
        )

    print('finished searching files for timestamps, created dict for timestamp matches with filepaths')

    granule_groups, insitu_indices = _build_granule_groups(best_matches)

    

    print('finished creating granule groups dict, fpaths with connected in-situ datapoint')

    results = []
    print('starting looping throgh granules to find spatial matchups')

    for fpath, insitu_indices in granule_groups.items():
        group_results = _process_granule_group(
            fpath=fpath,
            insitu_indices=insitu_indices,
            best_matches=best_matches,
            insitu_lat=insitu_lat,
            insitu_lon=insitu_lon,
            in_situ_value=in_situ_value,
            is_MODIS=is_MODIS,
            target_wls=target_wls,
            flag_mask=flag_mask,
            min_valid_pixels=min_valid_pixels,
            in_situ_timestamps = in_situ_timestamps
        )
        results.extend(group_results)

    return pd.DataFrame(results)


# ── Top-level pipeline runner (optional convenience wrapper) ─────────────
def run_matchup_pipeline(
    in_situ_path,
    L2_granule_directory,
    time_threshold_min=180,
    spatial_threshold_km=2,
    target_wls=[547, 667],
    is_MODIS=False,
    flag_mask=None,
    min_valid_pixels=5,
    moving=False,
    sat_match_filepath=None,
    delete_no_match_sat_files=False,
    output_directory='3x3matchups.csv'
):



    in_situ = pd.read_csv(in_situ_path)
    in_situ = in_situ[~in_situ['datetime'].isna()]
    in_situ = in_situ.reset_index()

    for row in in_situ.itertuples():
        if row.datetime.endswith('Z'):
            print(row)

    in_situ_timestamps = in_situ['datetime']
    in_situ_value = in_situ['PIC_mmolC_m3']
    in_situ_lat = in_situ['lat']
    in_situ_lon = in_situ['lon']

    df = matchup3x3(
        L2_granule_directory,
        in_situ_timestamps,
        in_situ_lat,
        in_situ_lon,
        in_situ_value,
        time_threshold_min,
        spatial_threshold_km,
        target_wls=target_wls,
        is_MODIS=is_MODIS,
        flag_mask=flag_mask,
        min_valid_pixels=min_valid_pixels,
        delete_no_match_sat_files=delete_no_match_sat_files,
        sat_match_filepath=sat_match_filepath,
        moving=moving
    )
    valid_files = _load_valid_granules(L2_granule_directory)

    datetimes_sat = _extract_sat_datetimes(valid_files)
   

    csv_name = f'matchups_{datetimes_sat[0]}_{datetimes_sat[-1]}.csv'
    output_csv = os.path.join(output_directory, csv_name)

    df.to_csv(output_csv)
    print(f'csv saved to {output_csv}')

    return df