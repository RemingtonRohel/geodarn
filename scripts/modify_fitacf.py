import os
import glob
import argparse
import pydarnio

from geodarn import geolocation, extract_records

def main(directory, out_dir, pattern, tx_site, rx_site):
    files = glob.glob(f'{directory}/{pattern}')
    print(files)
    for f in files:
        outname = out_dir + '/' + os.path.basename(f).strip('.fitacf') + '.adjusted.fitacf'
        print(f'{f} -> {outname}')
        if not os.path.isfile(outname):
            sdarn_read = pydarnio.SDarnRead(f)
            records = sdarn_read.read_fitacf()

            geo_records = []
            for i, rec in enumerate(records):
                print(f'\r{i / len(records) * 100:.1f}', flush=True, end='')
                result = geolocation.adjust_fitacf_in_place(rec, rx_site, tx_site)
                geo_records.append(result)
            print()
            writer = pydarnio.SDarnWrite(geo_records, outname)
            writer.write_fitacf(outname)




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
