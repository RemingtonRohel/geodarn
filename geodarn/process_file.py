import argparse

import numpy as np

# Imports from local files
import geolocation as gl
import file_ops
import extract_records as extraction
from utils import formats


def process_fitacf_file(infile, outfile, tx_site, rx_site):
    """
    Reads in a fitacf file and processes it into a geolocated outfile.

    Parameters
    ----------
    infile: str
        Path to fitacf file.
    outfile: str
        Path to output geolocated file.
    tx_site: str
        Three-letter radar code for the transmitter site. Lowercase letters.
    rx_site: str
        Three-letter radar code for the receiver site. Lowercase letters.
    """
    print(f'Reading file {infile}')
    records = file_ops.read_fitacf(infile)
    print(f'Grouping {len(records)} records by timestamp')
    merged_records = extraction.merge_simultaneous_records(records)
    print(f'Geolocating scatter in all {len(merged_records)} merged records')
    geo_records = []

    for rec in merged_records:
        result = gl.geolocate_record(rec, rx_site, tx_site)
        if result is not None:
            geo_records.append(result)

    data = formats.Data.create_from_records(geo_records)
    info = formats.Info(date=np.array([geo_records[0]['timestamp'].year,
                                       geo_records[0]['timestamp'].month,
                                       geo_records[0]['timestamp'].day], dtype=int),
                        experiment_cpid = records[0]['cp'],
                        experiment_name=records[0]['combf'],
                        tx_site_name=tx_site,
                        rx_site_name=rx_site,
                        rx_freq = records[0]['tfreq'])
    print(info)
    print(data)
    container = formats.Container(info=info, data=data)

    container.show()
    print(f'Writing results to file {outfile}')
    container.dataclass_to_hdf5()

    # file_ops.write_geographic_scatter(geo_records, outfile, rx_site, tx_site)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Path to input file.')
    parser.add_argument('outfile', help='Path to output file.')
    parser.add_argument('rx_site', help='Three-letter radar code for rx site')
    parser.add_argument('tx_site', help='Three-letter radar code for tx site')
    args = parser.parse_args()

    process_fitacf_file(args.infile, args.outfile, args.tx_site, args.rx_site)
