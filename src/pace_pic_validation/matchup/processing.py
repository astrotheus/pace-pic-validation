import pandas as pd
import os
import shutil 


def copy_granules(df, src_dir, dest_dir, granule_col='granule'):
    """
    Copy granule files listed in a DataFrame column from a source to a destination directory.

    For each granule name in `df[granule_col]`, this function:
      - Treats the value as a filename (strips any directory components).
      - Looks for the file in `src_dir`.
      - If found, copies it into `dest_dir` (overwriting if it already exists).
      - Counts how many files were copied and how many were not found.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing granule filenames.
    src_dir : str
        Source directory where granule files are expected.
    dest_dir : str
        Destination directory to copy granule files into.
    granule_col : str, default 'granule'
        Column name in `df` that holds the granule filenames.

    Prints a summary of how many files were copied and how many were missing.
    """
    moved = 0
    not_found = 0
    for granule in df[granule_col]:
        # Ensure we only have the filename, not a path
        granule = str(granule)
        http = granule.startswith('<')
        print(http)
        if http:
            print(f'skipping {granule}')
            continue


        fname = os.path.basename(str(granule))
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dest_dir, fname)

        if os.path.isfile(src):
            if not os.path.exists(dst):
                shutil.copy(src, dest_dir)  # dest can be directory
                moved += 1
            else:
                # already there; still remove from src if you really want "move"
                shutil.copy(src, dest_dir)
                moved += 1
        else:
            not_found += 1

    print(f"Copied: {moved}, Not found in {src_dir}: {not_found}")

def CL2algorithm(rrs547,rrs667, old = True ):
    cl2 = rrs547 - rrs667

    if old:
        PIC = 0.4579 * cl2 -0.0006 #this is in mol/m3

    else: 
        PIC = 79.446*(cl2**2.160) # this is in ? 
        
    return PIC