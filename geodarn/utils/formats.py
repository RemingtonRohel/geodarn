"""
Dataclass containers for ICEBEAR-3D data processing and HDF5 data packing.
"""
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
import numpy as np
import h5py
import datetime
import collections

from utils import parse_hdw


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
    date: np.ndarray #((3,), dtype=int) = field(
        # default=np.asarray([0, 0, 0], dtype=int),
        # metadata={'type': 'int32',
        #           'units': 'None',
        #           'shape': (3,),
        #           'version': __version__,
        #           'created': __created__,
        #           'description': 'starting date of the data contained within'})
    experiment_name: str #= field(
        # default=None,
        # metadata={'type': 'str',
        #           'units': 'None',
        #           'shape': None,
        #           'version': __version__,
        #           'created': __created__,
        #           'description': 'name of experiment ran (ex; normalscan, twofsound)'})
    experiment_cpid: int #= field(
        # default=None,
        # metadata={'type': 'int32',
        #           'units': 'None',
        #           'shape': None,
        #           'version': __version__,
        #           'created': __created__,
        #           'description': 'control program id of experiment ran'})
    # comment: str = field(
    #     default=None,
    #     metadata={'type': 'str',
    #               'units': 'None',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'comment included in the experiment'})
    rx_freq: float #= field(
        # default=None,
        # metadata={'type': 'float32',
        #           'units': 'hertz',
        #           'shape': None,
        #           'version': __version__,
        #           'created': __created__,
        #           'description': 'radar receiver frequency in Hz'})
    tx_site_name: str #= field(
        # default=None,
        # metadata={'type': 'str',
        #           'units': 'None',
        #           'shape': None,
        #           'version': __version__,
        #           'created': __created__,
        #           'description': 'name of the transmitter site'})
    # tx_site_lat_lon: np.ndarray((2,), dtype=float) = field(
        # default_factory=np.ndarray,
        # metadata={'type': 'float32',
        #           'units': 'degree',
        #           'shape': (2,),
        #           'version': __version__,
        #           'created': __created__,
        #           'description': '[latitude, longitude] global North-Easting coordinates of the transmitter site in degrees'})
    # tx_heading: float = field(
        # default_factory=float,
        # metadata={'type': 'float32',
        #           'units': 'degree',
        #           'shape': None,
        #           'version': __version__,
        #           'created': __created__,
        #           'description': 'transmitter array boresight pointing direction in degrees East of North'})
    rx_site_name: str #= field(
        # default=None,
        # metadata={'type': 'str',
        #           'units': 'None',
        #           'shape': None,
        #           'version': __version__,
        #           'created': __created__,
        #           'description': 'name of the receiver site'})
    # rx_site_lat_lon: np.ndarray((2,), dtype=float) = field(
    #     default_factory=np.ndarray,
    #     metadata={'type': 'float32',
    #               'units': 'degree',
    #               'shape': (2,),
    #               'version': __version__,
    #               'created': __created__,
    #               'description': '[latitude, longitude] global North-Easting coordinates of the receiver site in degrees'})
    # rx_heading: float = field(
    #     default_factory=float,
    #     metadata={'type': 'float32',
    #               'units': 'degree',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'receiver array boresight pointing direction in degrees East of North'})
    rx_sample_rate: float = field(
        default=3333.33333,
        metadata={'type': 'float32',
                  'units': 'hertz',
                  'shape': None,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the sample rate at the receiver in Hz'})
    date_created: str = field(
        default=__created__,
        metadata={'type': 'str',
                  'units': 'None',
                  'shape': 19,
                  'version': __version__,
                  'created': __created__,
                  'description': 'the date and time this file was generated'})

    def __post_init__(self):
        rx_hdw = parse_hdw.Hdw.read_hdw_file(self.rx_site_name)
        tx_hdw = parse_hdw.Hdw.read_hdw_file(self.tx_site_name)

        self.rx_site_lat_lon = np.array(rx_hdw.location[:2])
        self.rx_heading = rx_hdw.boresight_direction
        self.tx_site_lat_lon = np.array(tx_hdw.location[:2])
        self.tx_heading = tx_hdw.boresight_direction


@dataclass
class Data:
    time: np.ndarray
    location: np.ndarray
    power_db: np.ndarray
    velocity: np.ndarray
    velocity_dir: np.ndarray
    spectral_width: np.ndarray
    groundscatter: np.ndarray
    # time: np.ndarray = field(
    #     # default=np.array([], dtype=np.float32),
    #     metadata={'type': 'float32',
    #               'units': 'second',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'start time of each sample in seconds since epoch'})
    # location: np.ndarray = field(
    #     # default=np.array([], dtype=np.float32),
    #     metadata={'type': 'float32',
    #               'units': 'degrees',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'array of geographic (latitude, longitude) of each data point'})
    # power_db: np.ndarray = field(
    #     # default=np.array([], dtype=np.float32),
    #     metadata={'type': 'float32',
    #               'units': 'dB',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'array of powers in dB'})
    # velocity: np.ndarray = field(
    #     # default=np.array([], dtype=np.float32),
    #     metadata={'type': 'float32',
    #               'units': 'meters per second',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'array of velocity magnitudes'})
    # velocity_dir: np.ndarray = field(
    #     # default=np.array([], dtype=np.float32),
    #     metadata={'type': 'float32',
    #               'units': 'degrees',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'array of velocity directions in degrees East of North'})
    # spectral_width: np.ndarray = field(
    #     # default=np.array([], dtype=np.float32),
    #     metadata={'type': 'float32',
    #               'units': 'meters per second',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'array of spectral widths'})
    # groundscatter: np.ndarray = field(
    #     # default=np.array([], dtype=np.int8),
    #     metadata={'type': 'int8',
    #               'units': 'None',
    #               'shape': None,
    #               'version': __version__,
    #               'created': __created__,
    #               'description': 'groundscatter flag for each point'})
    time_slices: np.ndarray = field(
        init=False)#,
        # metadata={'type': 'int',
        #           'units': 'None',
        #           'shape': None,
        #           'version': __version__,
        #           'created': __created__,
        #           'description': '(start, stop) indices for each unique time'})

    def __post_init__(self):
        times, counts = np.unique(self.time, return_counts=True)
        time_slices = np.empty((len(times), 2), dtype=np.int32)
        end_indices = np.cumsum(counts)
        time_slices[:, 1] = end_indices
        time_slices[1:, 0] = end_indices[:-1]
        time_slices[0, 0] = 0
        self.time_slices = time_slices

    @classmethod
    def create_from_records(cls, records):
        """
        Instantiates the Data class with the contents of records.

        Parameters
        ----------
        records: list
            List of geolocated record dictionaries.

        Returns
        -------
        Data
            Data object with the relevant parameters from records
        """
        lists = collections.defaultdict(list)
        for rec in records:
            lists['time'].append([rec['timestamp'].timestamp()] * len(rec['v']))
            lists['location'].append(rec['scatter_location'])
            lists['power_db'].append(rec['p_l'])
            lists['velocity'].append(rec['v'])
            lists['velocity_dir'].append(rec['look_dir'])
            lists['spectral_width'].append(rec['w_l'])
            lists['groundscatter'].append(rec['gsct'])

        t = np.concatenate(lists['time'], dtype=np.float32)
        print(type(t))
        print(t.shape)
        return Data(time=t,
                    location=np.concatenate(lists['location']),
                    power_db=np.concatenate(lists['power_db']),
                    velocity=np.concatenate(lists['velocity']),
                    velocity_dir=np.concatenate(lists['velocity_dir']),
                    spectral_width=np.concatenate(lists['spectral_width']),
                    groundscatter=np.concatenate(lists['groundscatter']))
        Info(date=np.array([records[0]['timestamp'].year,
                            records[0]['timestamp'].month,
                            records[0]['timestamp'].day], dtype=int),
             experiment_cpid=records[0]['cp'],
             experiment_name=records[0]['combf'],
             tx_site_name=tx_site,
             rx_site_name=rx_site,
             rx_freq=records[0]['tfreq'])


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
    date: np.ndarray = field(
        metadata={'group': 'info',
                  'units': 'None'})
    experiment_name: str = field(
        metadata={'group': 'info',
                  'units': 'None'})
    experiment_cpid: int = field(
        metadata={'group': 'info',
                  'units': 'None'})
    rx_freq: float = field(
        metadata={'group': 'info',
                  'units': 'kHz'})
    tx_site_name: str = field(
        metadata={'group': 'info',
                  'units': 'None'})
    rx_site_name: str = field(
        metadata={'group': 'info',
                  'units': 'None'})
    time: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'seconds'})
    location: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'degrees'})
    power_db: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'dB'})
    velocity: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'm/s'})
    velocity_dir: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'degrees'})
    spectral_width: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'm/s'})
    groundscatter: np.ndarray = field(
        metadata={'group': 'data',
                  'units': 'None'})
    rx_sample_rate: float = field(
        default=3333.33333,
        metadata={'group': 'info',
                  'units': 'Hz'})
    date_created: str = field(
        default=__created__,
        metadata={'group': 'info',
                  'units': 'None'})
    time_slices: np.ndarray = field(
        init=False,
        metadata={'group': 'data',
                  'units': 'None'})
    rx_site_lat_lon: np.ndarray = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'degrees'})
    rx_heading: float = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'degrees'})
    tx_site_lat_lon: np.ndarray = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'degrees'})
    tx_heading: float = field(
        init=False,
        metadata={'group': 'info',
                  'units': 'degrees'})

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


    # info: Info = field(default_factory=Info, init=False)
    # data: Data = field(default_factory=Data, init=False)
    # dev: Dev = field(default_factory=Dev)
    # conf: Config = field(default_factory=Config)

    # def __repr__(self):
    #     return f'info: {self.info!r}\n' \
    #            f'data: {self.data!r}\n' #\
    #            # f'dev: {self.dev!r}\n' \
    #            # f'dev: {self.conf!r}'

    # def show(self):
    #     msg = f'{"=" * 200}\n' \
    #           f'{"Dataset":^30} | ' \
    #           f'{"Units":^20} | ' \
    #           f'{"Type":^15} | ' \
    #           f'{"Shape":^15} | ' \
    #           f'{"Ver":^5} | ' \
    #           f'{"Created":^20} | ' \
    #           f'{"    Description":<}\n' \
    #           f'{"=" * 200}\n'
    #     for x in fields(self.info):
    #         msg += f'{"info." + x.name:<30} | \n' \
    #                # f'{x.metadata["units"]:^20} | ' \
    #                # f'{x.metadata["type"]:^15} | ' \
    #                # f'{str(x.metadata["shape"]):^15} | ' \
    #                # f'{x.metadata["version"]:^5} | ' \
    #                # f'{x.metadata["created"]:^20} | ' \
    #                # f'{x.metadata["description"]:<}\n'
    #     for x in fields(self.data):
    #         msg += f'{"data." + x.name:<30} | \n' \
    #                # f'{x.metadata["units"]:^20} | ' \
    #                # f'{x.metadata["type"]:^15} | ' \
    #                # f'{str(x.metadata["shape"]):^15} | ' \
    #                # f'{x.metadata["version"]:^5} | ' \
    #                # f'{x.metadata["description"]:<}\n'
    #     # for x in fields(self.dev):
    #     #     msg += f'{"dev."+x.name:<30} | ' \
    #     # f'{x.metadata["units"]:^20} | ' \
    #     # f'{x.metadata["type"]:^15} | ' \
    #     # f'{str(x.metadata["shape"]):^15} | ' \
    #     # f'{x.metadata["version"]:^5} | ' \
    #     # f'{x.metadata["description"]:<}\n'
    #     return msg

    @classmethod
    def create_from_records(cls, records, tx_site, rx_site):
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
                         experiment_name=records[0]['combf'],
                         tx_site_name=tx_site,
                         rx_site_name=rx_site,
                         rx_freq=records[0]['tfreq'])

    def dataclass_to_hdf5(self, outfile):
        f = h5py.File(outfile, 'w')

        for x in fields(self):
            key = x.name
            path = f'{x.metadata.get("group")}/{key}'
            value = getattr(self, key)
            if type(value) is str:
                value = np.asarray(value, dtype='S')
            else:
                value = np.asarray(value)
            f.create_dataset(path, data=value)
            for k, v in x.metadata.items():
                if k != 'group':
                    f[path].attrs[k] = v

        f = h5py.File(outfile, 'r')
        print(f.items())

        def _print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f'\t{key}: {val}')
            return None

        f.visititems(_print_attrs)

        return

    @staticmethod
    def hdf5_to_dataclass(infile: str):
        pass
        return


if __name__ == '__main__':
    d = Container()
    print(d.show())

