import argparse

# Imports from local files
import pydarnio

import geolocation as gl
from geodarn import formats, extract_records as extraction


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
    sdarn_read = pydarnio.SDarnRead(infile)
    records = sdarn_read.read_fitacf()
    print(f'Grouping {len(records)} records by timestamp')
    merged_records = extraction.merge_simultaneous_records(records)
    print(f'Geolocating scatter in all {len(merged_records)} merged records')
    geo_records = []

    for rec in merged_records:
        result = gl.geolocate_record(rec, rx_site, tx_site)
        if result is not None:
            geo_records.append(result)

    container = formats.Container.create_located_from_records(geo_records, tx_site, rx_site)

    print(f'Writing results to file {outfile}')
    container.dataclass_to_hdf5(outfile)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Path to input file.')
    parser.add_argument('outfile', help='Path to output file.')
    parser.add_argument('rx_site', help='Three-letter radar code for rx site')
    parser.add_argument('tx_site', help='Three-letter radar code for tx site')
    args = parser.parse_args()

    process_fitacf_file(args.infile, args.outfile, args.tx_site, args.rx_site)
