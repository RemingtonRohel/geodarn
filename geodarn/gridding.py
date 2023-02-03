import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import plotting
import file_ops


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
        lon_divs.append(np.linspace(0, 360, num_lons[i] + 1))
        grid_mask[i, num_lons[i]+1:, :] = True     # Mask out all entries past the maximum

        lon_centers = lon_divs[i][:-1] + (lon_divs[i][1] - lon_divs[i][0])/2        # Center of each cell in longitude
        grid_centers[i, :len(lon_centers), 0] = lat_centers[i]
        grid_centers[i, :len(lon_centers), 1] = lon_centers

    grid = np.ma.array(grid_centers, mask=grid_mask)

    return lat_divs_bottom, lon_divs, grid


def create_grid_records(geo_records, lat_min=50, lat_width=1.0, hemisphere='north'):
    lat_divs_bottom, lon_divs, grid = create_grid(lat_min, lat_width, hemisphere)

    grid_dict = {}
    grid_dict['min_lat'] = lat_divs_bottom[0]
    grid_dict['lat_width'] = lat_width
    grid_dict['bins'] = []





if __name__ == '__main__':
    lat_divs_bottom, lon_divs, grid = create_grid(lat_min=55, lat_width=1.0)
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=-100))
    ax.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
    ax.coastlines(zorder=5, alpha=0.2)
    ax.gridlines()

    for lat_idx in range(grid.shape[0]):
        ax.scatter(x=grid[lat_idx, :, 1], y=grid[lat_idx, :, 0], transform=ccrs.PlateCarree(), color='k', alpha=0.5,
                   s=2)

    geo_records = file_ops.read_geodarn_file('/home/remington/geodarn/geodarn_test.hdf5')

    timestamps = sorted(list(geo_records['records'].keys()))
    rx_site = geo_records['records'][timestamps[0]]['rx_site']
    tx_site = geo_records['records'][timestamps[0]]['tx_site']
    site_ids = [rx_site]
    if rx_site != tx_site:
        site_ids.append(tx_site)

    # Plot the data on a map
    record = geo_records['records'][timestamps[0]]
    plotting.plot_single_param_from_scan(fig, ax, record, 'power', label='Power [dB]', site_ids=site_ids)

    plt.show()
    plt.close()
