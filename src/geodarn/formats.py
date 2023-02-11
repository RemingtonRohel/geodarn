"""
Dataclass containers for ICEBEAR-3D data processing and HDF5 data packing.
"""
import os
import re
from dataclasses import dataclass, field, fields
import numpy as np
import h5py
import datetime
import collections

from geodarn import parse_hdw, gridding


def version():
    here = os.path.abspath(os.path.dirname(__file__))
    regex = "(?<=__version__..\s)\S+"
    with open(os.path.join(here, '__init__.py'), 'r', encoding='utf-8') as f:
        text = f.read()
    match = re.findall(regex, text)
    return str(match[0].strip("'"))


def created(mode='str'):
    now = datetime.datetime.utcnow()
    if mode == 'str':
        return now.strftime('%Y-%m-%d %H:%M:%S')
    elif mode == 'arr':
        return np.array([now.year, now.month, now.day, now.hour, now.minute, now.second], dtype=int)


__version__ = version()
__created__ = created()


@dataclass
class Container:
    date: np.ndarray = field(
        metadata={'group': 'info',
                  'units': 'None',
                  'description': 'Date as [year, month, day] for when the data is from'})
    comment: str = field(
        metadata={'group': 'info',
                  'units': 'None',
                  'description': 'Any extra information about the file'})
    experiment_cpid: int = field(
        metadata={'group': 'info',
                  'units': 'None',
                  'description': 'Control program ID of the experiment'})
    rx_freq: float = field(
        metadata={'group': 'info',
                  'units': 'kHz',
                  'description': 'Operating frequency of the receiver radar'})
    tx_site_name: str = field(
        metadata={'group': 'info',
                  'units': 'None',
                  'description': 'Three-letter code of the transmitting radar'})
    rx_site_name: str = field(
        metadata={'group': 'info',
                  'units': 'None',
                  'description': 'Three-letter code of the receiving radar'})
    time: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'seconds',
                  'description': 'Array of size [num_points] of timestamps'})
    location: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'degrees',
                  'description': 'Array of [lat, lon] locations'})
    power_db: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'dB',
                  'description': 'Array of size [num_points] of the scatter power'})
    velocity: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'm/s',
                  'description': 'Array of size [num_points] of the scatter velocity'})
    velocity_dir: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'degrees',
                  'description': 'Array of size [num_points] of velocity directions CW of North'})
    spectral_width: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'm/s',
                  'description': 'Array of size [num_points] of spectral width'})
    groundscatter: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'None',
                  'description': 'Array of size [num_points] of flags for groundscatter'})
    date_created: str = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'None',
                  'description': 'Date when this file was created'})
    time_slices: np.ndarray = field(
        init=False,
        metadata={'group': 'data',
                  'units': 'None',
                  'description': 'Indices corresponding to unique times in the "time" array'})
    rx_site_lat_lon: np.ndarray = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'degrees',
                  'description': 'Location of receiving radar as [lat, lon]'})
    rx_heading: float = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'degrees',
                  'description': 'Forward look direction of receiving radar in degrees CW of North'})
    tx_site_lat_lon: np.ndarray = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'degrees',
                  'description': 'Location of transmitting radar as [lat, lon]'})
    tx_heading: float = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'degrees',
                  'description': 'Forward look direction of transmitting radar in degrees CW of North'})
    location_idx: np.ndarray = field(
        init=False,
        metadata={'group': 'data',
                  'units': 'None',
                  'optional': 'True',
                  'description': 'Index into first dimension of location array for grid file'})

    def __post_init__(self):
        times, counts = np.unique(self.time, return_counts=True)
        time_slices = np.empty((len(times), 2), dtype=np.int32)
        end_indices = np.cumsum(counts)
        time_slices[:, 1] = end_indices
        time_slices[1:, 0] = end_indices[:-1]
        time_slices[0, 0] = 0
        self.time_slices = time_slices

        rx_hdw = parse_hdw.Hdw.read_hdw_file(self.rx_site_name)
        tx_hdw = parse_hdw.Hdw.read_hdw_file(self.tx_site_name)
        self.rx_site_lat_lon = np.array(rx_hdw.location[:2])
        self.rx_heading = rx_hdw.boresight_direction
        self.tx_site_lat_lon = np.array(tx_hdw.location[:2])
        self.tx_heading = tx_hdw.boresight_direction

        self.date_created = __version__

    @staticmethod
    def create_located_from_records(records, tx_site, rx_site):
        """
        Instantiates the Data class with the contents of records.

        Parameters
        ----------
        records: list
            List of geolocated record dictionaries.
        tx_site: str
            Three-letter code of transmitting radar site.
        rx_site: str
            Three-letter code of receiving radar site.

        Returns
        -------
        Container
            Container object with the relevant parameters from records
        """
        lists = collections.defaultdict(list)
        for rec in records:
            lists['time'].append(np.repeat(np.float64(rec['timestamp'].timestamp()), len(rec['v'])))
            lists['location'].append(rec['scatter_location'])
            lists['power_db'].append(rec['p_l'])
            lists['velocity'].append(rec['v'])
            lists['velocity_dir'].append(rec['look_dir'])
            lists['spectral_width'].append(rec['w_l'])
            lists['groundscatter'].append(rec['gsct'])

        return Container(time=np.concatenate(lists['time']),
                         location=np.concatenate(lists['location']),
                         power_db=np.concatenate(lists['power_db']),
                         velocity=np.concatenate(lists['velocity']),
                         velocity_dir=np.concatenate(lists['velocity_dir']),
                         spectral_width=np.concatenate(lists['spectral_width']),
                         groundscatter=np.concatenate(lists['groundscatter']),
                         date=np.array([records[0]['timestamp'].year,
                                        records[0]['timestamp'].month,
                                        records[0]['timestamp'].day], dtype=int),
                         experiment_cpid=records[0]['cp'],
                         comment=records[0]['combf'],
                         tx_site_name=tx_site,
                         rx_site_name=rx_site,
                         rx_freq=records[0]['tfreq'])

    @staticmethod
    def create_gridded_from_located(located, **kwargs):
        """
        Instantiates the Data class with the contents of records.

        Parameters
        ----------
        located: Container
            Located dataclass
        **kwargs: dict
            supported values are 'lat_min', 'lat_width', 'hemisphere' for determining the grid.

        Returns
        -------
        Container
            Container object with the relevant parameters from records
        """

        idx_in_grid, grid = gridding.create_grid_records(located, lat_min=kwargs.get('lat_min', 50.0),
                                                         lat_width=kwargs.get('lat_width', 1.0),
                                                         hemisphere=kwargs.get('hemisphere', 'north'))

        gridded = Container(time=located.time,
                            location=grid,
                            power_db=located.power_db,
                            velocity=located.velocity,
                            velocity_dir=located.velocity_dir,
                            spectral_width=located.spectral_width,
                            groundscatter=located.groundscatter,
                            date=located.date,
                            experiment_cpid=located.experiment_cpid,
                            comment=located.comment,
                            tx_site_name=located.tx_site_name,
                            rx_site_name=located.rx_site_name,
                            rx_freq=located.rx_freq)
        gridded.location_idx = idx_in_grid

        return gridded

    def dataclass_to_hdf5(self, outfile):
        with h5py.File(outfile, 'w') as f:
            for x in fields(self):
                key = x.name
                path = f'{x.metadata.get("group")}/{key}'
                try:
                    value = getattr(self, key)
                except AttributeError:
                    continue
                if type(value) is str:
                    value = np.asarray(value, dtype='S')
                else:
                    value = np.asarray(value)
                if value.size == 1:
                    compression = None
                else:
                    compression = 'gzip'
                f.create_dataset(path, data=value, compression=compression)
                for k, v in x.metadata.items():
                    if k != 'group':
                        f[path].attrs[k] = v

    @classmethod
    def hdf5_to_dataclass(cls, infile: str):
        init_fields = {}
        after_init_fields = {}

        with h5py.File(infile, 'r') as f:
            for x in fields(cls):
                group = x.metadata.get('group')
                if not x.init:
                    if x.metadata.get('optional', 'False') == 'True' and x.name in f[f'{group}'].keys():
                        after_init_fields[x.name] = f[f'{group}/{x.name}'][()]
                    continue
                if issubclass(x.type, str):
                    temp_array = np.char.decode(f[f'{group}/{x.name}'])
                    init_fields[x.name] = np.array_str(temp_array)
                else:
                    init_fields[x.name] = f[f'{group}/{x.name}'][()]
        container = Container(**init_fields)

        for k, v in after_init_fields.items():
            setattr(container, k, v)

        return container
