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
import glob
import argparse

import geodarn


def main(directory, out_dir, pattern, tx_site, rx_site):
    files = glob.glob(f'{directory}/{pattern}')
    print(files)
    for f in files:
        outname = out_dir + '/' + os.path.basename(f).strip('.fitacf') + '.located.hdf5'
        print(f'{f} -> {outname}')
        if not os.path.isfile(outname):
            geodarn.create_located_from_fitacf(f, outname, tx_site, rx_site)
        grid_name = outname.strip('.located.hdf5') + '.gridded.hdf5'
        print(f'{outname} -> {grid_name}')
        if not os.path.isfile(grid_name):
            located = geodarn.formats.Container.from_hdf5(outname)
            gridded = geodarn.formats.create_gridded_from_located(located)
            gridded.to_hdf5(grid_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory with fitacf files')
    parser.add_argument('rx_site', help='Three-letter radar code of receiver site')
    parser.add_argument('tx_site', help='Three-letter radar code of transmitter site')
    parser.add_argument('--pattern', default='*.fitacf', help='Pattern to search for files in directory')
    parser.add_argument('--out-dir', help='Directory to store resultant files in', type=str)
    args = parser.parse_args()

    if not args.out_dir:
        out_directory = args.directory
    else:
        out_directory = args.out_dir
    main(args.directory, out_directory, args.pattern, args.tx_site, args.rx_site)
