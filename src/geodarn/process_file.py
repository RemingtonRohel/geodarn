import argparse

from geodarn import formats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Path to input file.')
    parser.add_argument('outfile', help='Path to output file.')
    parser.add_argument('rx_site', help='Three-letter radar code for rx site')
    parser.add_argument('tx_site', help='Three-letter radar code for tx site')
    args = parser.parse_args()

    formats.create_located_from_fitacf(args.infile, args.outfile, args.tx_site, args.rx_site)
