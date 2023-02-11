import os
import copy

import shapely.geometry
import numpy as np
import cartopy.crs as ccrs
import cartopy.geodesic
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

import parse_hdw
from utils.constants import sites
import geolocation as gl


def azimuthal_lines(site, min_range, max_range, beam_width=3.24, elevation=0.0):
    """
    Plots the equidistant curves for the two radar sites, and the FOV of each site.

    :param site: Three-letter identifier of transmit site
    :param min_range: Distance to center of first range, in km.
    :param max_range: Distance to far edge of furthest range, in km.
    :param beam_width: Angular separation between beams, in degrees.
    :param elevation: Degrees above horizontal
    """
    site_data = parse_hdw.Hdw.read_hdw_file(site)

    az_lines = []

    # Compute the azimuthal divisions of the site FOV
    for i in range(-8, 9):
        tx_bearing = site_data.boresight_direction + gl.azimuth_from_elevation(elevation, i * beam_width)
        ranges = np.linspace(min_range, max_range, 100) * 1e3  # meters
        tx_radial_line = cartopy.geodesic.Geodesic().direct((site_data.location[1], site_data.location[0]),
                                                            tx_bearing, ranges)
        tx_geom = shapely.geometry.LineString(tx_radial_line)
        az_lines.append(tx_geom)

    return az_lines


def plot_geolocated_lines(axis, param, start_points, end_points, values):
    """
    Plots the equidistant curves for the two radar sites, and the FOV of each site.

    :param axis: matplotlib axis to plot on
    :param param: parameter to plot (used to get colormap)
    :param start_points: Array of (longitudes, latitudes) of the scattering locations
    :param end_points: Array of end points for each scattering velocity vector
    :param values: Array of values for coloring lines.
    """
    cmap, norm = get_cmap_and_norm(param)
    for i in range(start_points.shape[0]):
        axis.plot([start_points[i, 0], end_points[i, 0]], [start_points[i, 1], end_points[i, 1]],
                  color=cmap(norm(values[i])), linewidth=1.0, transform=ccrs.Geodetic())


def plot_geolocated_scatter(axis, param, locations, values):
    """
    Plots the equidistant curves for the two radar sites, and the FOV of each site.

    :param axis: matplotlib axis to plot on
    :param param: parameter to plot (used to get colormap)
    :param locations: Array of (longitudes, latitudes) of the scattering locations
    :param values: Array of values for coloring lines.
    """
    cmap, norm = get_cmap_and_norm(param)
    axis.scatter(locations[:, 0], locations[:, 1], c=cmap(norm(values)), alpha=0.5, s=5, transform=ccrs.PlateCarree())


def generate_bistatic_axes(site_ids, dims, center=(-110, 60), lon_extent=(-140, 0), lat_extent=(55, 65),
                           size=(12, 12), **kwargs):
    """
    Build a matplotlib figure for use in plotting bistatic data.

    :param site_ids:    list of three-letter identifiers of for sites to have FoVs included on plot.
    :param dims:        tuple of (num_rows, num_cols) for setting up subplots
    :param size:        tuple for figure size, in inches.
    :param center:      tuple of (longitude, latitude) for setting up the center of the orthographic projection
    :param lon_extent:  tuple of (lon, lon) defining the longitudinal extent of the plot.
    :param lat_extent:  tuple of (lat, lat) defining the latitudinal extent of the plot.
    :param **kwargs:    Any other keyword arguments which will be passed to plt.subplots()
    """
    ids = [parse_hdw.Hdw.read_hdw_file(s) for s in site_ids]

    # Set up the plot
    ot = ccrs.Orthographic(center[0], center[1])
    fig, axes = plt.subplots(dims[0], dims[1], figsize=size, subplot_kw={'projection': ot},
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05}, **kwargs)
    xs, ys, zs = ot.transform_points(ccrs.PlateCarree(), np.array(lon_extent), np.array(lat_extent)).T

    def configure_axes(ax):
        ax.set_xlim(xs)
        ax.set_ylim(ys)
        # ax.set_facecolor('grey')
        ax.coastlines(zorder=5, alpha=0.2)
        ax.gridlines()

        # Mark the sites
        for site, site_id in zip(ids, site_ids):
            ax.plot(site.location[1], site.location[0], markersize=5, color='blue', marker='o',
                    transform=ccrs.PlateCarree())
            platecarree_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
            tx_transform = offset_copy(platecarree_transform, units='dots', x=site['hshift'], y=site['vshift'])
            ax.text(site.location[1], site.location[0], f'\\textbf{{{site_id.upper()}}}', verticalalignment='bottom',
                    horizontalalignment='left', transform=tx_transform)

    if isinstance(axes, np.ndarray):
        for axis in axes.flatten():
            configure_axes(axis)
    else:
        configure_axes(axes)

    return fig, axes


def get_cmap_and_norm(param):
    """
    Returns a colormap and norm for a given parameter.

    :param param: Parameter in question. One of 'v', 'p_l', 'w_l', 'elv', or 'h_v'
    """
    params = ['power', 'velocity', 'spectral_width', 'lobe']#, 'groundscatter']
    if param not in params:
        raise ValueError(f'Unknown parameter {param}:\n'
                         f'Supported parameters are {params}')

    value_ranges = {'power': [0, 40],
                    'velocity': [-400, 400],
                    'spectral_width': [0, 1000],
                    'lobe': [0, 7]
                    }
    colors = {
        'power': 'plasma',
        'velocity': matplotlib.colors.LinearSegmentedColormap.from_list('pydarn_velocity',
                                                                 ['darkred', 'r', 'pink', 'b', 'darkblue']),
        'spectral_width': 'viridis',
        'lobe': matplotlib.colors.ListedColormap(['grey', 'red', 'blue', 'green', 'yellow', 'orange', 'purple'])
    }

    cmap = copy.copy(plt.cm.get_cmap(colors[param]))
    cmap.set_bad(color='k', alpha=1.0)
    norm = matplotlib.colors.Normalize(vmin=value_ranges[param][0], vmax=value_ranges[param][1])

    return cmap, norm


def add_colorbar(fig, axis, param, **kwargs):
    """
    Add a colorbar to the axis of a figure.

    :param fig: Figure containing the axis
    :param axis: Axis to add colorbar to
    :param param: Parameter to use for obtaining colormap and limits
    :param kwargs: Any other kwargs to pass to colorbar instantiation.
    """
    if param == 'groundscatter':
        ticks = [0.25, 0.75]
    elif param == 'lobe':
        ticks = [(1+2*i)/2 for i in range(8)]
    else:
        ticks = None
    cmap, norm = get_cmap_and_norm(param)

    extend = {'power': 'max',
              'velocity': 'both',
              'spectral_width': 'max',
              'lobe': 'neither',
              'groundscatter': 'neither'
              }
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, extend=extend[param], ticks=ticks,
                        **kwargs)
    if param == 'groundscatter':
        cbar.ax.set_yticklabels(['Ionosphere', 'Ground'])
    elif param == 'lobe':
        cbar.ax.set_yticklabels([i for i in range(8)])


def add_fov_boundaries(axis, site):
    """
    Adds FOV boundaries for a site to a matplotlib axis.

    :param axis: matplotlib axis
    :param site: three-letter identifier for radar site
    """
    fov = azimuthal_lines(site, 0, 180 + 45 * 75, beam_width=3.24)
    axis.add_geometries((fov[0], fov[-1]), crs=ccrs.Geodetic(), facecolor='none', edgecolor='black', linewidth=0.5,
                        alpha=1)

    return axis


def plot_single_param_from_scan(fig, ax, record, idx_slice, param, label=None, site_ids=(), stem=True, colorbar=True):

    plot_values = getattr(record, param)[idx_slice]
    locations = getattr(record, 'location')[idx_slice]

    if param == 'velocity' and stem:
        scale_value = 250  # Ratio of geographic distance in meters of plotted vector to LOS velocity value in m/s
        end_points = cartopy.geodesic.Geodesic().direct(locations, getattr(record, 'velocity_dir')[idx_slice],
                                                        scale_value * plot_values)
        plot_geolocated_lines(ax, param, locations, end_points[:, :2], plot_values)

    plot_geolocated_scatter(ax, param, locations, plot_values)

    if colorbar:
        add_colorbar(fig, ax, param, shrink=0.5)
    if label is not None:
        ax.set_title(label)

    # Add FOV boundaries
    for site_id in site_ids:
        add_fov_boundaries(ax, site_id)


def plot_scans_on_map(infile, plot_dir, prefix=''):
    """
    Takes a fitacf file and plots the power, velocity, spectral width, and elevation
    for all times. Each plot is saved separately in plot_dir with the format {prefix}YYYYMMDD-HH:MM:SS.ffffff.png
    """
    geo_records = file_ops.read_geodarn_file(infile)

    timestamps = sorted(list(geo_records['records'].keys()))
    rx_site = geo_records['records'][timestamps[0]]['rx_site']
    tx_site = geo_records['records'][timestamps[0]]['tx_site']
    site_ids = [rx_site]
    if rx_site == tx_site:
        lon_extent = sites[rx_site]['lon_extent']
        lat_extent = sites[rx_site]['lat_extent']
    else:
        site_ids.append(tx_site)
        lon_extent = (-131, -30)
        lat_extent = (60, 75)

    for tstamp in timestamps:
        print(tstamp)

        if os.path.isfile(f'{plot_dir}/{prefix}{tstamp}.png'):
            continue

        # Plot the data on a map
        fig, axes = generate_bistatic_axes(site_ids, (2, 2), lon_extent=lon_extent, lat_extent=lat_extent,
                                           size=(8, 8))

        record = geo_records['records'][tstamp]
        plot_single_param_from_scan(fig, axes[0, 0], record, 'power', label='Power [dB]', site_ids=site_ids)
        plot_single_param_from_scan(fig, axes[0, 1], record, 'velocity', label='Velocity [m/s]', site_ids=site_ids,
                                    stem=False)
        plot_single_param_from_scan(fig, axes[1, 0], record, 'spectral_width', label='Spectral Width [m/s]',
                                    site_ids=site_ids)
        # plot_single_param_from_scan(fig, axes[1, 1], record, 'elv', label='Elevation [deg]', site_ids=site_ids)
        fig.tight_layout()

        fig.suptitle(tstamp)
        plt.savefig(f'{plot_dir}/{prefix}{tstamp}.png')
        plt.close()
