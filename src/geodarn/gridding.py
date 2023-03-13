import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def create_grid(lat_min=50, lat_width=1., hemisphere='north'):
    """
    Creates a grid of near-equal area cells in latitude and longitude.

    Parameters
    ----------
    lat_min: int
        Lower latitude boundary for the grid, in degrees. Default 50
    lat_width: float
        Cell with in degrees of latitude. Default 1.0
    hemisphere: str
        Which hemisphere the grid is for. Default 'north', 'south' also accepted. If 'south', grid goes from -90 to
        -lat_min.

    Returns
    -------

    """
    lat_divs_bottom = np.arange(lat_min, 90, lat_width)
    if hemisphere == 'south':
        lat_divs_bottom = np.flip(lat_divs_bottom * -1)
    elif hemisphere != 'north':
        raise ValueError(f'Unrecognized hemisphere {hemisphere}')

    lat_centers = lat_divs_bottom + lat_width/2

    cos_lat = np.cos(np.deg2rad(lat_centers))
    num_lons = np.int32(np.rint(360.0 / (lat_width / cos_lat)))

    max_num_lons = int(np.max(num_lons))

    grid_mask = np.ma.make_mask_none((len(lat_centers), max_num_lons, 2))
    grid_centers = np.zeros((len(lat_centers), max_num_lons, 2))

    lon_divs = []
    for i in range(num_lons.shape[0]):
        lon_divs.append(np.linspace(-180, 180, num_lons[i] + 1))
        grid_mask[i, num_lons[i]+1:, :] = True     # Mask out all entries past the maximum

        lon_centers = lon_divs[i][:-1] + (lon_divs[i][1] - lon_divs[i][0])/2        # Center of each cell in longitude
        grid_centers[i, :len(lon_centers), 1] = lat_centers[i]
        grid_centers[i, :len(lon_centers), 0] = lon_centers

    grid = np.ma.array(grid_centers, mask=grid_mask)

    return lat_divs_bottom, lon_divs, grid


def create_grid_records(located, lat_min=50, lat_width=1.0, hemisphere='north'):
    """
    Grids all points in located and creates a Gridded dataclass output.

    Parameters
    ----------
    located: Container
        Dataclass storing the geolocated points
    lat_min: int
        Lower latitude boundary for the grid.
    lat_width: float
        Latitudinal width of bins for the grid.
    hemisphere: str
        Either 'north' or 'south'.

    Returns
    -------

    """
    lat_divs_bottom, lon_divs, grid = create_grid(lat_min, lat_width, hemisphere)
    num_lons = grid.shape[1]
    grid = grid.reshape(-1, 2)   # flatten so that lats iterate more slowly than lons
    idx_in_grid = np.ones(located.location.shape[0], dtype=np.int32) * -1
    non_nan_locations = np.argwhere(~np.isnan(located.location[:, 0]))
    min_lat = np.min(located.location[non_nan_locations, 1])
    max_lat = np.max(located.location[non_nan_locations, 1])
    min_lon = np.min(located.location[non_nan_locations, 0])
    max_lon = np.max(located.location[non_nan_locations, 0])

    lat_indices = np.argwhere(np.logical_and(lat_divs_bottom > min_lat - lat_width, lat_divs_bottom < max_lat))

    # Loop through the latitude bins with data
    for lat_idx in lat_indices:
        lat = lat_divs_bottom[lat_idx]
        matching_points_lat = np.logical_and(lat < located.location[:, 1], located.location[:, 1] < lat + lat_width)
        matching_indices_lat = np.argwhere(matching_points_lat)
        lon_divs_for_lat = lon_divs[lat_idx[0]]
        lon_indices = np.argwhere(np.logical_and(min_lon < lon_divs_for_lat, lon_divs_for_lat < max_lon))

        # Loop through the longitude bins with data
        for lon_idx in lon_indices:
            matching_points_lon = np.logical_and(
                lon_divs_for_lat[lon_idx] < located.location[matching_points_lat, 0],
                located.location[matching_points_lat, 0] < lon_divs_for_lat[lon_idx + 1])
            # Record the indices into grid for each point
            idx_in_grid[matching_indices_lat[np.argwhere(matching_points_lon)]] = lat_idx[0] * num_lons + lon_idx[0]

    return idx_in_grid, grid


if __name__ == '__main__':
    from geodarn import formats
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=-100))
    ax.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
    ax.coastlines(zorder=5, alpha=0.4)
    ax.gridlines()

    located = formats.Container.from_hdf5('/data/special_experiments/202301/bistatic/inv/fitacf/20230110.1800.00.inv.located.hdf5')
    gridded = formats.create_gridded_from_located(located)
    tx_site = gridded.tx_site_name
    rx_site = gridded.rx_site_name
    site_ids = [tx_site, rx_site]

    indices = gridded.time_slices[1]
    slice_obj = slice(indices[0], indices[1])
    
    # Plot the data on a map
    # plotting.plot_single_param_from_scan(fig, ax, container, slice_obj, 'velocity', label='Velocity [m/s]',
    #                                      site_ids=site_ids, stem=True)

    # indices_in_grid, grid = create_grid_records(gridded)

    single_scan_indices = gridded.location_idx[slice_obj]

    for idx in single_scan_indices:
        ax.scatter(x=gridded.location[idx, 0], y=gridded.location[idx, 1], transform=ccrs.PlateCarree(),
                   color='k', alpha=0.2, s=2)

    plt.show()
    plt.close()

    # gridded.to_hdf5('/home/remington/repos/geodarn/20230110.1800.00.inv.gridded.hdf5')
    print('Done')
