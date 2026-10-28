

import pandas as pd

def units_to_PIC_mmol_m3(df, start_unit, start_column, end_column, caco3=False):
    """
    Convert a pandas column to mmol C / m3.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    start_unit : str
        Unit of the input column.
    start_column : str
        Name of the input column.
    end_column : str
        Name of the output column to create.
    caco3 : bool, default False
        If True, mass units are assumed to be CaCO3 equivalents.
        If False, mass units are assumed to be carbon.

    Returns
    -------
    pandas.DataFrame
        Copy of df with a new converted column.
    """
    M_caco3 = 100.0869
    M_c = 12.0107
    M = M_caco3 if caco3 else M_c

    conversion_factors = {
        'mol/m3': 1000,
        'mmol/m3': 1,
        'mol/L': 1,      # 1 mol/L = 1000 mol/m3 = 1,000,000 mmol/m3
        'mmol/L': 1 / 1000,     # 1 mmol/L = 1000 mmol/m3
        'mg/m3': 1 / M,
        'mg/L': 1 / M,   # same as g/m3
        'ug/L': 1 / M,
        'g/m3': 1000 / M,
    }

    if start_unit not in conversion_factors:
        raise ValueError(f"Unsupported unit: {start_unit}")

    out = df.copy()
    out[start_column] = pd.to_numeric(out[start_column], errors='coerce')
    out[end_column] = out[start_column] * conversion_factors[start_unit]
    
    return out


def timestamp_matchup(df1, df2, time_column1, time_column2, time_tolerance, columns_to_drop):
    """
    Match up two DataFrames based on timestamps within a specified tolerance.

    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame.
    df2 : pandas.DataFrame
        Second DataFrame.
    time_column1 : str
        Name of the timestamp column in df1.
    time_column2 : str
        Name of the timestamp column in df2.
    time_tolerance : pd.Timedelta
        Maximum allowed difference between timestamps for a match.
    columns_to_drop : list of str
        Columns to drop from the merged DataFrame.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with matched timestamps.
    """
    df1[time_column1] = pd.to_datetime(df1[time_column1])
    df2[time_column2] = pd.to_datetime(df2[time_column2])

    merged_df = pd.merge_asof(
        df1.sort_values(time_column1),
        df2.sort_values(time_column2),
        left_on=time_column1,
        right_on=time_column2,
        direction='nearest',
        tolerance=time_tolerance
    )

    merged_df.drop(columns=columns_to_drop, inplace=True)
    
    return merged_df


