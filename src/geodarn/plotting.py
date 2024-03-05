# This file is part of geodarn.
#
# geodarn is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# geodarn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with geodarn.
# If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2024 Remington Rohel

import copy
import warnings
from datetime import datetime
import argparse
import os

import numpy as np
import cartopy.crs as ccrs
import cartopy.geodesic
import matplotlib
import matplotlib.pyplot as plt
import pydarn

from geodarn import parse_hdw, formats
from geodarn.utils.constants import sites


def plot_geolocated_lines(ax, param, start_points, end_points, values, **kwargs):
    """
    Plots the equidistant curves for the two radar sites, and the FOV of each site.

    :param ax: matplotlib axis to plot on
    :param param: parameter to plot (used to get colormap)
    :param start_points: Array of (longitudes, latitudes) of the scattering locations
    :param end_points: Array of end points for each scattering velocity vector
    :param values: Array of values for coloring lines.
    """
    cmap, norm = get_cmap_and_norm(param, **kwargs)
    for i in range(start_points.shape[0]):
        ax.plot([start_points[i, 0], end_points[i, 0]], [start_points[i, 1], end_points[i, 1]],
                color=cmap(norm(values[i])), linewidth=1.0, transform=ccrs.Geodetic())


def plot_geolocated_scatter(axis, param, locations, values, **kwargs):
    """
    Plots the equidistant curves for the two radar sites, and the FOV of each site.

    :param axis: matplotlib axis to plot on
    :param param: parameter to plot (used to get colormap)
    :param locations: Array of (longitudes, latitudes) of the scattering locations
    :param values: Array of values for coloring lines.
    """
    cmap, norm = get_cmap_and_norm(param, **kwargs)
    s = kwargs.get('s', 20)
    alpha = kwargs.get('alpha', 1.0)
    axis.scatter(locations[:, 0], locations[:, 1], c=cmap(norm(values)),
                 # alpha=alpha,
                 s=s,
                 linewidths=0.0,
                 transform=ccrs.PlateCarree())


def generate_bistatic_axes(site_ids, dims, center=(-110, 60), lon_extent=(-140, 0), lat_extent=(55, 65),
                           size=(12, 12), gridlines=False, **kwargs):
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
        ax.coastlines(zorder=5, alpha=0.2)
        ax.gridlines(visible=gridlines)

    if isinstance(axes, np.ndarray):
        for axis in axes.flatten():
            configure_axes(axis)
    else:
        configure_axes(axes)

    return fig, axes


def get_cmap_and_norm(param, **kwargs):
    """
    Returns a colormap and norm for a given parameter.

    :param param: Parameter in question. One of 'v', 'p_l', 'w_l', 'elv', or 'h_v'
    """
    params = ['power_db', 'velocity', 'spectral_width', 'lobe']#, 'groundscatter']
    if param not in params:
        raise ValueError(f'Unknown parameter {param}:\n'
                         f'Supported parameters are {params}')

    value_ranges = {'power_db': [0, 40],
                    'velocity': [-1000, 1000],
                    'spectral_width': [0, 1000],
                    'lobe': [-7.5, 7.5]
                    }
    colors = {
        'power_db': 'plasma',
        'velocity': matplotlib.colors.LinearSegmentedColormap.from_list('pydarn_velocity',
                                                                        ['darkred', 'r', 'pink', 'b', 'darkblue']),
        'spectral_width': 'viridis',
        'lobe': 'coolwarm'
    }
    if 'colormap' not in kwargs or kwargs['colormap'] is None:
        colormap = colors[param]
    else:
        colormap = kwargs['colormap']

    cmap = copy.copy(plt.cm.get_cmap(colormap))
    if param == 'lobe':
        old_cmap = list(map(cmap, range(256)))
        cmap = cmap.from_list('newcmap', old_cmap[:110:18] + [(0, 0, 0, 0.4)] + old_cmap[146::18], N=15)

    value_range = [kwargs.get('vmin', value_ranges[param][0]), kwargs.get('vmax', value_ranges[param][1])]

    norm = matplotlib.colors.Normalize(vmin=value_range[0], vmax=value_range[1])

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
        ticks = [i for i in range(-7, 8)]
    else:
        ticks = None

    cmap, norm = get_cmap_and_norm(param, **kwargs)
    kwargs.pop('colormap', None)

    extend = {'power_db': 'max',
              'velocity': 'both',
              'spectral_width': 'max',
              'lobe': 'neither',
              'groundscatter': 'neither'
              }
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, extend=extend[param], ticks=ticks,
                        **kwargs)
    if param == 'groundscatter':
        cbar.ax.set_yticklabels(['Ionosphere', 'Ground'])


def add_fov_boundaries(axis, site):
    """
    Adds FOV boundaries for a site to a matplotlib axis.

    :param axis: matplotlib axis
    :param site: three-letter identifier for radar site
    """
    hdw_data = parse_hdw.Hdw.read_hdw_file(site)
    pydarn.Fan.plot_fov(hdw_data.station_id,
                        None,  # datetime, shouldn't matter for this projection.
                        ax=axis,
                        ccrs=ccrs,
                        coords=pydarn.Coords.GEOGRAPHIC,
                        projs=pydarn.Projs.GEO)
    return axis


def plot_single_param_from_scan(fig, ax, record, idx_slice, param, label=None, site_ids=(), stem=True, colorbar=True,
                                **kwargs):
    warnings.simplefilter('ignore', UserWarning)
    plot_values = getattr(record, param)[idx_slice]
    locations = getattr(record, 'location')[idx_slice]

    if param == 'velocity' and stem:
        scale_value = 250  # Ratio of geographic distance in meters of plotted vector to LOS velocity value in m/s
        end_points = cartopy.geodesic.Geodesic().direct(locations, getattr(record, 'velocity_dir')[idx_slice],
                                                        scale_value * plot_values)
        plot_geolocated_lines(ax, param, locations, end_points[:, :2], plot_values, **kwargs)

    plot_geolocated_scatter(ax, param, locations, plot_values, **kwargs)

    if colorbar:
        add_colorbar(fig, ax, param, shrink=0.7, **kwargs)
    if label is not None:
        ax.set_title(label)

    # Add FOV boundaries
    for site_id in site_ids:
        add_fov_boundaries(ax, site_id)


def plot_scans_on_map(infile, plot_dir, prefix=''):
    """
    Takes a located file and plots the power, velocity, spectral width, and elevation
    for all times. Each plot is saved separately in plot_dir with the format {prefix}YYYY-MM-DD_HH-MM-SS.ffffff.png
    """
    data = formats.Container.from_hdf5(infile)
    rx_site = data.rx_site_name
    tx_site = data.tx_site_name

    site_ids = [rx_site]
    if rx_site == tx_site:
        lon_extent = sites[rx_site]['lon_extent']
        lat_extent = sites[rx_site]['lat_extent']
    else:
        site_ids.append(tx_site)
        lon_extent = (-131, -30)
        lat_extent = (60, 75)

    time_slices = data.time_slices

    for i in range(len(time_slices)):
        tstamp = datetime.fromtimestamp(data.time[time_slices[i, 0]]).strftime('%Y-%m-%d_%H-%M-%S.%f')
        slice_obj = slice(time_slices[i, 0], time_slices[i, 1])

        print(tstamp)

        # Plot the data on a map
        fig, axes = generate_bistatic_axes(site_ids, (1, 3), lon_extent=lon_extent, lat_extent=lat_extent,
                                           size=(15, 5))

        plot_single_param_from_scan(fig, axes[0], data, slice_obj, 'power_db', label='Power [dB]',
                                    site_ids=site_ids)
        plot_single_param_from_scan(fig, axes[1], data, slice_obj, 'velocity', label=f'{tstamp}\n\nVelocity [m/s]',
                                    site_ids=site_ids, stem=True)
        plot_single_param_from_scan(fig, axes[2], data, slice_obj, 'spectral_width', label='Spectral Width [m/s]',
                                    site_ids=site_ids)

        plt.savefig(f'{plot_dir}/{prefix}{tstamp}.png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='File to plot scans from')
    parser.add_argument('plot_dir', help='Directory to save plots in. Default is same as directory', default='')
    args = parser.parse_args()

    if args.plot_dir == '':
        plot_dir = os.path.dirname(args.infile)
    else:
        plot_dir = args.plot_dir

    # infile = '/data/special_experiments/202303/full_fov/cly/20230306.1330-1930.cly.located.hdf5'
    # plot_dir = '/data/special_experiments/202303/full_fov/cly/scans/'
    plot_scans_on_map(args.infile, plot_dir)
