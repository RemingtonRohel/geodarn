"""
Dataclass containers for ICEBEAR-3D data processing and HDF5 data packing.
"""
import os
import re
from dataclasses import dataclass, field, fields
import numpy as np
import h5py
import datetime

from . import parse_hdw


def version():
    here = os.path.abspath(os.path.dirname(__file__))
    regex = "(?<=__version__..\s)\S+"
    with open(os.path.join(here, '../__init__.py'), 'r', encoding='utf-8') as f:
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
class Info:
    date_created: str = field(
        default=__created__,
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 19,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the date and time this file was generated'})
    date: np.ndarray((3,), dtype=int) = field(
        default=np.asarray([0, 0, 0], dtype=int),
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': (3,),
                  'version': __version__,
                  'created': __created__,
                  'description': 'starting date of the data contained within'})
    experiment_name: str = field(
        default=None,
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'name of experiment ran (ex; normalscan, twofsound)'})
    experiment_cpid: int = field(
        default=None,
        metadata={'type': 'int32',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'control program id of experiment ran'})
    comment: str = field(
        default=None,
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'comment included in the experiment'})
    rx_freq: float = field(
        default=None,
        metadata={'type': 'float32',
                  'units': 'hertz',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'radar receiver frequency in Hz'})
    tx_site_name: str = field(
        default=None,
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'name of the transmitter site'})
    tx_site_lat_lon: np.ndarray((2,), dtype=float) = field(
        default_factory=np.ndarray,
        metadata={'type': 'float32',
                  'units': 'degree',
                  'shape': (2,),
                  'version': __version__,
                  'created': __created__,
                  'description': '[latitude, longitude] global North-Easting coordinates of the transmitter site in degrees'})
    tx_heading: float = field(
        default_factory=float,
        metadata={'type': 'float32',
                  'units': 'degree',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'transmitter array boresight pointing direction in degrees East of North'})
    rx_site_name: str = field(
        default=None,
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'name of the receiver site'})
    rx_site_lat_lon: np.ndarray((2,), dtype=float) = field(
        default_factory=np.ndarray,
        metadata={'type': 'float32',
                  'units': 'degree',
                  'shape': (2,),
                  'version': __version__,
                  'created': __created__,
                  'description': '[latitude, longitude] global North-Easting coordinates of the receiver site in degrees'})
    rx_heading: float = field(
        default_factory=float,
        metadata={'type': 'float32',
                  'units': 'degree',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'receiver array boresight pointing direction in degrees East of North'})
    rx_sample_rate: float = field(
        default=3333.33333,
        metadata={'type': 'float32',
                  'units': 'hertz',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the sample rate at the receiver in Hz'})

    def __post_init__(self):
        rx_hdw = parse_hdw.Hdw.read_hdw_file(self.rx_site_name)
        tx_hdw = parse_hdw.Hdw.read_hdw_file(self.tx_site_name)

        self.rx_site_lat_lon = np.array(rx_hdw.location[:2])
        self.rx_heading = rx_hdw.boresight_direction
        self.tx_site_lat_lon = np.array(tx_hdw.location[:2])
        self.tx_heading = tx_hdw.boresight_direction


@dataclass
class Data:
    time: np.ndarray((), dtype=float) = field(
        default=None,
        metadata={'type': 'float32',
                  'units': 'second',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'start time of each sample in seconds since epoch'})
    time_slices: np.ndarray((), dtype=int) = field(
        default_factory=np.ndarray,
        metadata={'type': 'int',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': '(start, stop) indices for each unique time'})
    location: np.ndarray((), dtype=float) = field(
        default=None,
        metadata={'type': 'float32',
                  'units': 'degrees',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'array of geographic (latitude, longitude) of each data point'})
    power_db: np.ndarray((), dtype=float) = field(
        default=None,
        metadata={'type': 'float32',
                  'units': 'dB',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'array of powers in dB'})
    velocity: np.ndarray((), dtype=float) = field(
        default=None,
        metadata={'type': 'float32',
                  'units': 'meters per second',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'array of velocity magnitudes'})
    velocity_dir: np.ndarray((), dtype=float) = field(
        default=None,
        metadata={'type': 'float32',
                  'units': 'degrees',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'array of velocity directions in degrees East of North'})
    spectral_width: np.ndarray((), dtype=float) = field(
        default=None,
        metadata={'type': 'float32',
                  'units': 'meters per second',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'array of spectral widths'})
    groundscatter: np.ndarray((), dtype=int) = field(
        default=None,
        metadata={'type': 'int',
                  'units': 'None',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'groundscatter flag for each point'}
    )

    def __post_init__(self):
        times, counts = np.unique(self.time, return_counts=True)
        self.time_slices = np.zeros((len(times), 2), dtype=np.int32)
        end_indices = np.cumsum(counts)
        self.time_slices[:, 1] = end_indices
        self.time_slices[1:, 0] = end_indices[:-1]

    # rf_distance: np.ndarray
    # snr_db: np.ndarray
    # doppler_shift: np.ndarray
    # spectra: np.ndarray
    # spectra_variance: np.ndarray
    # xspectra: np.ndarray
    # xspectra_variance: np.ndarray
    # latitude: np.ndarray
    # longitude: np.ndarray
    # altitude: np.ndarray
    # azimuth: np.ndarray
    # elevation: np.ndarray
    # slant_range: np.ndarray
    # velocity_azimuth: np.ndarray
    # velocity_elevation: np.ndarray
    # velocity_magnitude: np.ndarray
    # # Per second but for all range-Doppler bins
    # avg_spectra_noise: np.ndarray
    # spectra_noise: np.ndarray
    # xspectra_noise: np.ndarray
    # spectra_clutter_corr: np.ndarray
    # xspectra_clutter_corr: np.ndarray
    # data_flag: np.ndarray


# @dataclass(kw_only=True, slots=True)
# class Dev:
#     raw_elevation: np.ndarray
#     mean_jansky: np.ndarray
#     max_jansky: np.ndarray
#     validity: np.ndarray
#     classification: np.ndarray
#     azimuth_extent: np.ndarray
#     elevation_extent: np.ndarray
#     area: np.ndarray
#     doppler_spectra: np.ndarray
#
#
# @dataclass(kw_only=True, slots=True)
# class Config:
#     level0_dir: str = field(default='')
#     level1_dir: str = field(default='')
#     level2_dir: str = field(default='')
#     level3_dir: str = field(default='')
#     start_time: list[int] = field(default_factory=list)
#     stop_time: list[int] = field(default_factory=list)
#     step_time: list[int] = field(default_factory=list)


@dataclass
class Container:
    info: Info = field(default_factory=Info)
    data: Data = field(default_factory=Data)

    # dev: Dev = field(default_factory=Dev)
    # conf: Config = field(default_factory=Config)

    def __repr__(self):
        return f'info: {self.info!r}\n' \
               f'data: {self.data!r}\n' #\
               # f'dev: {self.dev!r}\n' \
               # f'dev: {self.conf!r}'

    def show(self):
        msg = f'{"=" * 200}\n' \
              f'{"Dataset":^30} | ' \
              f'{"Units":^20} | ' \
              f'{"Type":^15} | ' \
              f'{"Shape":^15} | ' \
              f'{"Ver":^5} | ' \
              f'{"Created":^20} | ' \
              f'{"    Description":<}\n' \
              f'{"=" * 200}\n'
        for x in fields(self.info):
            msg += f'{"info." + x.name:<30} | ' \
                   f'{x.metadata["units"]:^20} | ' \
                   f'{x.metadata["type"]:^15} | ' \
                   f'{str(x.metadata["shape"]):^15} | ' \
                   f'{x.metadata["version"]:^5} | ' \
                   f'{x.metadata["created"]:^20} | ' \
                   f'{x.metadata["description"]:<}\n'
        for x in fields(self.data):
            msg += f'{"data." + x.name:<30} | ' \
                   f'{x.metadata["units"]:^20} | ' \
                   f'{x.metadata["type"]:^15} | ' \
                   f'{str(x.metadata["shape"]):^15} | ' \
                   f'{x.metadata["version"]:^5} | ' \
                   f'{x.metadata["description"]:<}\n'
        # for x in fields(self.dev):
        #     msg += f'{"dev."+x.name:<30} | ' \
        # f'{x.metadata["units"]:^20} | ' \
        # f'{x.metadata["type"]:^15} | ' \
        # f'{str(x.metadata["shape"]):^15} | ' \
        # f'{x.metadata["version"]:^5} | ' \
        # f'{x.metadata["description"]:<}\n'
        return msg

    @classmethod
    def dataclass_to_hdf5(cls, path=''):
        f = h5py.File(path + 'geodarn_temp.hdf5', 'w')

        def loop(k, n=''):
            for a in fields(k):
                key = a.name
                try:
                    value = getattr(k, a.name)
                except AttributeError as err:
                    if a.name in ['info', 'data', 'dev']:
                        pass
                    else:
                        raise AttributeError(f'Attribute {a.name} has no Value')
                if a.type is type:
                    dset = a.name + '/'
                    loop(value, dset)
                else:
                    if n != '':
                        if type(value) is str:
                            value = np.asarray(value, dtype='S')
                        else:
                            value = np.asarray(value)
                        print(n + key)
                        f.create_dataset(n + key, data=value)
                        for kee, vaa in a.metadata.items():
                            f[n + key].attrs[kee] = vaa

        loop(cls)

        f = h5py.File(path + 'geodarn_temp.hdf5', 'r')
        print(f.items())

        def _print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f'\t{key}: {val}')
            return None

        f.visititems(_print_attrs)

        return

    def hdf5_to_dataclass(file: str):
        pass
        return


if __name__ == '__main__':
    d = Container()
    print(d.show())

