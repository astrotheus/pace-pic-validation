# src/pace_pic_validation/io/download.py
import os
import earthaccess


def list_oci_collections(instrument="OCI"):
    """
    List available short names for a given instrument (default: OCI).

    Returns
    -------
    list of str
        Short names of available collections.
    """
    collections = earthaccess.search_datasets(instrument=instrument)
    short_names = [c["umm"]["ShortName"] for c in collections]
    return short_names


def search_granules(short_name, start_date, end_date, granule_pattern="*", lower_left_lon = None, lower_left_lat=None, upper_right_lon=None, upper_right_lat=None):
    """
    Search PACE granules for a given product, date range, and filename pattern.

    Parameters
    ----------
    short_name : str
        Product short name, e.g. "PACE_OCI_L2_AOP_NRT"
    start_date, end_date : str
        ISO date strings, e.g. "2026-05-01"
    granule_pattern : str
        Wildcard pattern to filter granule names, e.g. "*AOP*"

    Returns
    -------
    list of earthaccess granule results
    """
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(start_date, end_date),
        granule_name=granule_pattern,
        bounding_box = (lower_left_lon, 
                        lower_left_lat, 
                        upper_right_lon, 
                        upper_right_lat)
    )
    print(f"Found {len(results)} granules")
    return results


def download_granules(results, local_path="data/raw", threads=1):
    """
    Download a list of earthaccess granule results to a local folder.

    Parameters
    ----------
    results : list
        Output from search_pace_granules()
    local_path : str
        Destination folder (default: data/raw, matches project structure)
    threads : int
        Number of parallel download threads

    Returns
    -------
    list of str
        Local file paths of downloaded granules
        
    """
    import earthaccess

    os.makedirs(local_path, exist_ok=True)

    earthaccess.login(persist=True)
    try:
        paths = earthaccess.download(results, local_path=local_path, threads=threads)
        print(f"Downloaded {len(paths)} files")
        return paths
    except Exception as e:
        print(f"Error: {e}")
        return []
    
import earthaccess

def find_collections(keyword=None, instrument=None, platform=None, daac=None, provider=None):
    """
    Universal collection finder for any NASA Earthdata mission.
    keyword: free-text search (e.g. 'ocean color', 'chlorophyll', 'sea ice')
    instrument: e.g. 'OCI', 'MODIS', 'VIIRS', 'ATLAS'
    platform: e.g. 'PACE', 'Aqua', 'Terra', 'Suomi NPP'
    daac: e.g. 'OB_DAAC', 'LAADS', 'LPDAAC', 'NSIDC'
    provider: alternative to daac, sometimes more reliable in CMR
    """
    kwargs = {}
    if keyword: kwargs["keyword"] = keyword
    if instrument: kwargs["instrument"] = instrument
    if platform: kwargs["platform"] = platform
    if daac: kwargs["daac"] = daac
    if provider: kwargs["provider"] = provider

    results = earthaccess.search_datasets(**kwargs)
    return [
        {
            "short_name": c["umm"]["ShortName"],
            "title": c["umm"].get("EntryTitle", ""),
            "platform": [p["ShortName"] for p in c["umm"].get("Platforms", [])],
        }
        for c in results
    ]

