from datetime import datetime
import collections

import numpy as np

import geodarn.utils.fitacf_format as fitacf


def extract_single_scan(fitacf_data, nearest_time):
    """
    Extracts records from fitacf data that are closer than 2 seconds to the given datetime.

    Parameters
    ----------
    fitacf_data: list[dict]
        List of FITACF record dictionaries.
    nearest_time: datetime
        Datetime to match against.

    Returns
    -------
    closest_records: list[dict]
        All records within 2 seconds of nearest_time.
    """
    closest_records = []
    for i, rec in enumerate(fitacf_data):
        rec_time = datetime(rec['time.yr'], rec['time.mo'], rec['time.dy'],
                            rec['time.hr'], rec['time.mt'], rec['time.sc'])
        if abs((nearest_time - rec_time).total_seconds()) < 2.0:
            closest_records.append(rec)

    return closest_records


def extract_records_by_beam(fitacf_data, beam):
    """
    Extracts only the records with a given beam number.

    Parameters
    ----------
    fitacf_data: list[dict]
        List of FITACF record dictionaries.
    beam: int
        Beam number to extract

    Returns
    -------
    valid_records: list[dict]
        All records for the given beam.
    """
    valid_records = []
    for i, rec in enumerate(fitacf_data):
        if rec['bmnum'] == beam:
            valid_records.append(rec)
    return valid_records


def extract_between_times(fitacf_data, start_time, end_time):
    """
    Extracts scans from fitacf data that are between the given datetimes.

    Parameters
    ----------
    fitacf_data: list[dict]
        List of FITACF record dictionaries.
    start_time: datetime
        Start of valid time interval.
    end_time: datetime
        End of valid time interval.

    Returns
    -------
    closest_records: list[dict]
        All records within with timestamps between start_time and end_time.
    """
    closest_records = []
    for i, rec in enumerate(fitacf_data):
        rec_time = datetime(rec['time.yr'], rec['time.mo'], rec['time.dy'],
                            rec['time.hr'], rec['time.mt'], rec['time.sc'])
        if start_time <= rec_time <= end_time:
            closest_records.append(rec)

    return closest_records


def group_records_by_timestamp(records):
    """
    Takes a list of records and groups records with matching timestamps.

    Parameters
    ----------
    records: list[dict]
        List of FITACF record dictionaries.

    Returns
    -------
    output_records: list[list[dict]]
        Nested list of records, where each inner list contains records with identical timestamps.
    """
    output_records = []
    current_group = [records[0]]
    current_time = datetime(records[0]['time.yr'], records[0]['time.mo'], records[0]['time.dy'],
                            records[0]['time.hr'], records[0]['time.mt'], records[0]['time.sc'])
    i = 1
    while i < len(records):
        rec = records[i]
        rec_time = datetime(rec['time.yr'], rec['time.mo'], rec['time.dy'],
                            rec['time.hr'], rec['time.mt'], rec['time.sc'])
        if rec_time == current_time:    # same scan, combine these records
            current_group.append(rec)
        else:                           # different scan, save the last list and start a new one
            output_records.append(current_group)
            current_group = [rec]
            current_time = rec_time
        i += 1

    return output_records


def merge_simultaneous_records(records):
    """
    Takes a list of records and merges records with matching timestamps.

    Parameters
    ----------
    records: list[dict]
        List of FITACF record dictionaries.

    Returns
    -------
    output_records: list[dict]
        List of records
    """
    grouped_records = group_records_by_timestamp(records)
    merged_records = []

    for group in grouped_records:
        merged_record = collections.defaultdict(list)

        for k in group[0].keys():
            # Add all fields unique to each record in the scan
            if k in fitacf.record_specific_vectors:
                for rec in group:
                    try:
                        merged_record[k].append(rec[k])
                    except KeyError:    # There was no data for the record
                        pass
                if len(merged_record[k]) != 0:
                    merged_record[k] = np.concatenate(merged_record[k])
                else:
                    merged_record[k] = np.array([])
            elif k in fitacf.record_specific_scalars:
                for rec in group:
                    try:
                        merged_record[k].append(np.ones(rec['slist'].shape) * rec[k])
                    except KeyError:
                        # There was no data for the record. Continue
                        pass
                if len(merged_record[k]) != 0:
                    merged_record[k] = np.concatenate(merged_record[k])
                else:
                    merged_record[k] = []

            # Add all fields common across the simultaneous records
            elif k in fitacf.scan_specific_scalars:
                merged_record[k] = group[0][k]
            elif k in fitacf.scan_specific_vectors:
                merged_record[k] = group[0][k]

            # Unknown quantity
            else:
                raise KeyError(f'Unknown record key {k}')

        merged_records.append(merged_record)

    return merged_records
