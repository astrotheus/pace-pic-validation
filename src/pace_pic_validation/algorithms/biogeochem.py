



def CL2algorithm(rrs547,rrs667, old = True ):
    cl2 = rrs547 - rrs667

    if old:
        PIC = 0.4579 * cl2 -0.0006 #this is in mol/m3

    else: 
        PIC = 79.446*(cl2**2.160) # this is in ? 
        
    return PIC




def chl_pic_oc6_pace_merged(ds, lat, lon):


        lats = ds["latitude"]
        lons = ds["longitude"]

        dlat = lats - lat
        dlon = lons - lon
        dist = np.sqrt(
            dlat**2 + (dlon * np.cos(np.deg2rad(lat)))**2
        )

        # Stack lat & lon into a single dimension
        stacked = dist.stack(loc=("latitude", "longitude"))

        # Flat index of minimum distance
        min_idx = int(stacked.argmin(dim="loc").values)

        # Get corresponding lat/lon indices via unravel
        n_lat = ds.sizes["latitude"]
        n_lon = ds.sizes["longitude"]

        idx_lat, idx_lon = np.unravel_index(min_idx, (n_lat, n_lon))

        # Extract the nearest pixel
        ds_pt = ds.isel(latitude=idx_lat, longitude=idx_lon)

        blue_wls = [412.0, 443.0, 490.0, 510.0]
        green_wls = [555.0, 678.0]

        rrs_spec = ds_pt["Rrs"]  # 1D in wavelength

        rrs_blue = rrs_spec.sel(wavelength=blue_wls, method="nearest")
        rrs_green = rrs_spec.sel(wavelength=green_wls, method="nearest")

        rrs_547 = rrs_spec.sel(wavelength=547, method="nearest")
        rrs_667 = rrs_spec.sel(wavelength=667, method="nearest")
        # OC6_PACE_OCI: max over blue, mean over green

    # OC6_PACE_OCI: max over blue, mean over green
        rrs_blue_max = float(rrs_blue.max())
        rrs_green_mean = float(rrs_green.mean())

        if rrs_green_mean <= 0:
            return np.nan

        MBR = rrs_blue_max / rrs_green_mean
        X = np.log10(MBR)

        a0, a1, a2, a3, a4 = 0.94297, -3.18493, 2.33682, -1.23923, 0.18697

        log_chl = a0 + a1*X + a2*X**2 + a3*X**3 + a4*X**4
        chl = 10**log_chl

        pic = CL2algorithm(rrs_547, rrs_667, old = True) *1000

        return chl, pic


def chl_pic_oc6_pace_granule(granule_path, lon, lat):
        """
        Compute a single chlorophyll-a value for a given (lon, lat)
        from a PACE OCI L2 granule using the OC6_PACE_OCI algorithm.

        Parameters
        ----------
        granule_path : str
            Path to the PACE L2 granule file.
        lon, lat : float
            Target longitude and latitude.

        Returns
        -------
        chl : float
            Chlorophyll-a concentration in mg m-3.
        """
        from pace_pic_validation.io.preprocessing import open_l2_minimal


        ds = open_l2_minimal(granule_path)


        # Distance-based nearest pixel (great-circle approximation)
        dlat = ds.latitude - lat
        dlon = ds.longitude - lon
        dist = np.sqrt(dlat**2 + (dlon * np.cos(np.deg2rad(lat)))**2)

        idx = dist.argmin(dim=["number_of_lines", "pixels_per_line"])

        ds_pt = ds.isel(
            number_of_lines=idx["number_of_lines"],
            pixels_per_line=idx["pixels_per_line"],
        )

        # Extract Rrs spectrum at this pixel: 1D array over wavelength
        rrs_spec = ds_pt["Rrs"]  # dims: (wavelength,)

        # Select blue and green bands (use nearest wavelengths)
        blue_wls = [412.0, 443.0, 490.0, 510.0]
        green_wls = [555.0, 678.0]

        rrs_blue = rrs_spec.sel(wavelength=blue_wls, method="nearest")
        rrs_green = rrs_spec.sel(wavelength=green_wls, method="nearest")

        # OC6_PACE_OCI: max over blue, mean over green
        rrs_blue_max = float(rrs_blue.max())
        rrs_green_mean = float(rrs_green.mean())
        rrs_547 = rrs_spec.sel(wavelength=547, method="nearest")
        rrs_667 = rrs_spec.sel(wavelength=677, method="nearest")

        pic = CL2algorithm(rrs_547, rrs_667, old = True)

        if rrs_green_mean <= 0:
            return np.nan

        MBR = rrs_blue_max / rrs_green_mean
        X = np.log10(MBR)

        a0, a1, a2, a3, a4 = 0.94297, -3.18493, 2.33682, -1.23923, 0.18697

        log_chl = a0 + a1*X + a2*X**2 + a3*X**3 + a4*X**4
        chl = 10**log_chl

        return chl, pic    