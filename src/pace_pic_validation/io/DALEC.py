def openDALEC(path):

    '''
    opens a Dalec pickle file and merges on datetime index. index is datetime, 
    while wavelengths and lat and lon are columns.
    
    '''

    import pandas as pd
    import pickle

    print(f'opening {path} as a pandas df')

    x = pickle.load(open(path, 'rb'))
    rrs = x['Rrs']
    lat = x['Ancillary']['LATITUDE']
    lon = x['Ancillary']['LONGITUDE'] 

    rrs.index = pd.to_datetime(rrs.index)
    lat.index = pd.to_datetime(lat.index)
    lon.index = pd.to_datetime(lon.index)
    df = pd.concat([rrs, lat, lon], axis=1, join="inner")

    return df

def xy_dalec(dalec, lat, lon):

    import numpy as np
    from sklearn.neighbors import BallTree, KDTree
    import pandas as pd

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

    dalec = dalec.drop(columns = ['LATITUDE', 'LONGITUDE', 'LATITUDE_rad', 'LONGITUDE_rad','chla_mg_m3','pic_mmol_m3' ])
    
    row = dalec.loc[datetime]

    x = dalec.columns
    y = row

    return x, y, index 


def chl_oc6_dalec(rrs):
        """
        rrs: dict-like with keys at or near:
            412, 443, 490, 510, 555, 678 (nm)
        returns chlorophyll-a in mg m-3 using OC6_PACE_OCI.
        """
        # rrs is a DataFrame, columns include 412, 443, 490, 510, 555, 678
        blue_cols = [412, 444, 490, 510]
        green_cols = [556, 678]

        rrs_blue_max = rrs[blue_cols].max(axis=1)          # Series, one value per row
        rrs_green_mean = rrs[green_cols].mean(axis=1)      # Series, one value per row


        MBR = rrs_blue_max / rrs_green_mean
        X = np.log10(MBR)

        a0, a1, a2, a3, a4 = 0.94297, -3.18493, 2.33682, -1.23923, 0.18697

        log_chl = a0 + a1*X + a2*X**2 + a3*X**3 + a4*X**4
        return 10**log_chl