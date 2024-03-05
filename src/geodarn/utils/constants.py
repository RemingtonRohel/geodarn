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

"""
Helpful constants for analyzing/processing SuperDARN FITACF files.
"""

sites = {'inv': {'lat': 68.414,
                 'lon': -133.772,
                 'boresight': 29.5,
                 'valign': 'center',
                 'halign': 'right',
                 'vshift': -70,
                 'hshift': 65,
                 'intf_in_front': 1,
                 'tdiff': 0.0,
                 'lon_extent': (-135, 0),
                 'lat_extent': (65, 65),
                 },
         'rkn': {'lat': 62.828,
                 'lon': -92.113,
                 'boresight': 5.7,
                 'valign': 'top',
                 'halign': 'left',
                 'vshift': 0,
                 'hshift': 38,
                 'intf_in_front': -1,
                 'tdiff': 0.042,
                 'lon_extent': (-131, -30),
                 'lat_extent': (60, 75),
                 },
         'cly': {'lat': 70.487,
                 'lon': -68.504,
                 'boresight': -55.62,
                 'valign': 'bottom',
                 'halign': 'left',
                 'vshift': 38,
                 'hshift': -38,
                 'intf_in_front': 1,
                 'tdiff': 0.0,
                 'lon_extent': (-131, -30),
                 'lat_extent': (60, 75),
                 },
         'pgr': {'lat': 53.980,
                 'lon': -122.590,
                 'boresight': -5.0,
                 'valign': 'top',
                 'halign': 'center',
                 'vshift': -10,
                 'hshift': 0,
                 'intf_in_front': -1,
                 'tdiff': 0.0,
                 'lon_extent': (-131, -30),
                 'lat_extent': (60, 75),
                 },
         'sas': {'lat': 52.160,
                 'lon': -106.530,
                 'boresight': 23.1,
                 'valign': 'top',
                 'halign': 'center',
                 'vshift': -10,
                 'hshift': 0,
                 'intf_in_front': -1,
                 'tdiff': 0.0,
                 'lon_extent': (-120, -20),
                 'lat_extent': (48, 60),
                 }
         }

