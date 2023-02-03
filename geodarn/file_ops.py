from datetime import datetime

import numpy as np
import h5py
import pydarnio

from utils import formats


def read_fitacf(infile: str):
    """
    Reads a fitacf file and returns the data within.

    Parameters
    ----------
    infile: str
        Path to the fitacf file.

    Returns
    -------
    list[dict]:
        List of records within the file.
    """
    sdarn_read = pydarnio.SDarnRead(infile)
    return sdarn_read.read_fitacf()


def write_geographic_scatter(records, outfile, rx_site, tx_site):
    """
    Writes a list of geolocated records to outfile. Saves as an HDF5 file.

    Parameters
    ----------
    records: list[dict]
        List of geolocated record dictionaries.
    outfile: str
        Path to file in which to save data.
    """
    with h5py.File(outfile, 'w') as f:
        first_record = records[0]

        # These fields should be fixed across the entirety of the file
        f.attrs['experiment_cpid'] = first_record['cp']
        f.attrs['station_id'] = first_record['stid']
        f.attrs['comment'] = first_record['combf']

        # These fields can vary record by record and so are stored within records.
        # Scalar values are given their own dataset so they can be easily combined with files from other radars
        # to generate grid files.
        for rec in records:
            start_time = datetime(rec['time.yr'], rec['time.mo'], rec['time.dy'],
                                  rec['time.hr'], rec['time.mt'], rec['time.sc'])

            group = f.create_group(start_time.strftime('%Y%m%d-%H:%M:%S.%f'))  # e.g. 20210101-18:30:01.906748
            group.attrs['rx_site'] = rx_site
            group.attrs['tx_site'] = tx_site

            # The following values are expected to be single values per record
            integration_time = group.create_dataset('integration_time', (1,), dtype='f4')
            num_sequences = group.create_dataset('num_sequences', (1,), dtype='i2')
            freq = group.create_dataset('freq', (1,), dtype='f4')

            integration_time[0] = np.float32(rec['intt.sc']) + np.float32(rec['intt.us']) * 1e-6
            num_sequences[0] = np.int16(rec['nave'])
            freq[0] = np.float32(rec['tfreq'])

            # The following values are expected to be arrays of values per record
            num_points = len(rec['lobe'])    # Should be the same for every dataset

            locations = group.create_dataset('locations', (num_points, 2), 'f4')
            velocity_dirs = group.create_dataset('velocity_dirs', (num_points,), 'f4')
            groundscatter_flag = group.create_dataset('groundscatter_flag', (num_points,), 'i1')
            lobe = group.create_dataset('lobe', (num_points,), 'i1')
            power = group.create_dataset('power', (num_points,), 'f4')
            velocity = group.create_dataset('velocity', (num_points,), 'f4')
            spectral_width = group.create_dataset('spectral_width', (num_points,), 'f4')

            locations[:, :] = rec['scatter_location']
            velocity_dirs[:] = rec['look_dir']
            groundscatter_flag[:] = np.int8(rec['gsct'])
            lobe[:] = rec['lobe']
            power[:] = rec['p_l']
            velocity[:] = rec['v']
            spectral_width[:] = rec['w_l']


def read_geodarn_file(infile):
    """
    Reads an HDF5 file containing geolocated SuperDARN data.

    Parameters
    ----------
    infile: str
        Path to file with geolocated SuperDARN data.

    Returns
    -------
    dict:
        Dictionary of attributes and datasets stored in the file.
    """
    with h5py.File(infile, 'r') as f:
        file_dict = {
            'experiment_cpid': f.attrs['experiment_cpid'],
            'station_id': f.attrs['station_id'],
            'comment': f.attrs['comment'],
            'records': {}
        }

        for rec in sorted(list(f.keys())):
            rec_dict = {}
            for k in f[rec].keys():
                rec_dict[k] = f[rec][k][()]     # load into memory
            for k in f[rec].attrs:
                rec_dict[k] = f[rec].attrs[k]
            file_dict['records'][rec] = rec_dict

    return file_dict
