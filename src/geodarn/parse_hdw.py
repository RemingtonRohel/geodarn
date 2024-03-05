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

import os
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Hdw:
    """
    Class for  storing information provided in standard SuperDARN hdw.dat files.
    """
    station_id: int
    status_code: int
    valid_from: datetime
    location: tuple
    boresight_direction: float
    offset: float
    beam_separation: float
    velocity_sign: int
    phase_sign: int
    tdiff_a: float
    tdiff_b: float
    intf_x_offset: float
    intf_y_offset: float
    intf_z_offset: float
    rx_rise_time: float
    rx_attenuation_db: float
    rx_attenuation_stages: int
    max_num_ranges: int
    max_num_beams: int

    @classmethod
    def read_hdw_file(cls, site):
        """
        Create a Hdw dataclass for the specified site. Reads the hdw.dat.[radar] file from hdw/

        Parameters
        ----------
        site: str
            Three-letter code for the radar site.
        """
        if len(site) != 3:
            raise ValueError(f'Invalid site code: {site}')
        site_name = site.lower()
        filename = f'{os.path.dirname(__file__)}/hdw/hdw.dat.{site_name}'
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Find the valid line
        i = -1
        line = lines[i]
        while line[0] == '#':
            i -= 1
            line = lines[i]

        fields = line.split()
        return Hdw(station_id=int(fields[0]),
                   status_code=int(fields[1]),
                   valid_from=datetime.strptime(f'{fields[2]} {fields[3]}', '%Y%m%d %H:%M:%S'),
                   location=(float(fields[4]), float(fields[5]), float(fields[6])),
                   boresight_direction=float(fields[7]),
                   offset=float(fields[8]),
                   beam_separation=float(fields[9]),
                   velocity_sign=int(fields[10]),
                   phase_sign=int(fields[11]),
                   tdiff_a=float(fields[12]),
                   tdiff_b=float(fields[13]),
                   intf_x_offset=float(fields[14]),
                   intf_y_offset=float(fields[15]),
                   intf_z_offset=float(fields[16]),
                   rx_rise_time=float(fields[17]),
                   rx_attenuation_db=float(fields[18]),
                   rx_attenuation_stages=int(fields[19]),
                   max_num_ranges=int(fields[20]),
                   max_num_beams=int(fields[21]))


if __name__ == '__main__':
    sas = Hdw.read_hdw_file('sas')
    print(sas)
