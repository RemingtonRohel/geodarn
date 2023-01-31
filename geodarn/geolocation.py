import numpy as np
import cartopy.geodesic

from utils import sites


def find_elevation(site_id, beam_dirs, freq_hz, phi0):
    """
    Finds the elevation given a horizontal beam direction and phase offset.

    Parameters
    ----------
    site_id: str
        Three-letter radar code, such as 'rkn' or 'sas'. Lowercase letters.
    beam_dirs: np.array
        Array of beam directions at zero elevation.
    freq_hz: float
        Frequency in Hz
    phi0: float
        Phase offset between main and interferometer arrays, in radians.
    """
    site = sites[site_id]   # hardware information from here

    c = 299792458
    wave_num = 2 * np.pi * freq_hz / c
    beam_dirs = np.deg2rad(beam_dirs)
    antenna_sep = 100   # meters between array midpoints

    cable_offset = -2 * np.pi * freq_hz * site['tdiff'] * 1e-6     # implement tdiff here if needed
    phase_diff_max = site['intf_in_front'] * wave_num * antenna_sep * np.cos(beam_dirs) + cable_offset

    psi_uncorrected = phi0 + 2 * np.pi * np.floor((phase_diff_max - phi0) / (2 * np.pi))

    if site['intf_in_front'] < 0:
        psi_uncorrected += 2 * np.pi

    psi = psi_uncorrected - cable_offset
    psi_kd = psi / (wave_num * antenna_sep)
    theta = np.cos(beam_dirs)*np.cos(beam_dirs) - psi_kd*psi_kd

    elevation = np.arcsin(np.sqrt(theta))
    # elevation[phi0 < 0.0] = 0.0
    # elevation[np.fabs(phi0) > 1.0] = 0.0

    return np.rad2deg(elevation)


def sidelobe_finder(beam_dirs, freq_hz, antenna_spacing_m=15.24, num_antennas=16):
    """
    Returns the first 7 sidelobes given a beam direction and frequency.

    Parameters
    ----------
    beam_dirs: np.array
        Array of main lobe directions. Shape [num_points]
    freq_hz: float
        Frequency in Hz.
    antenna_spacing_m: float
        Antenna spacing in meters. Default 15.24, for most SuperDARN radars.
    num_antennas: int
        Number of antennas in array. Included here for future-proofing, but only 16 antennas supported.

    Returns
    -------
    sidelobes: np.ndarray of shape [num_points, num_sidelobes]
        Array of sidelobe locations for all beam_dirs.
    """
    if num_antennas != 16:
        raise ValueError('Sidelobe finder only valid for 16 equally-spaced antennas.')

    c = 299792458   # speed of light

    # Hardcoded sidelobe locations in frequency and main-lobe direction independent coordinates for array of 16
    # equally-spaced antennas.
    qd = np.array([-1.4722, -1.2751, -1.0778, -0.8803, -0.6824, -0.4835, -0.2812,
                   0.2812,  0.4835,  0.6824,  0.8803,  1.0778,  1.2751,  1.4722],
                  dtype=np.float32)
    q = qd / antenna_spacing_m
    sin_beam = np.sin(np.deg2rad(beam_dirs))
    k = 2 * np.pi * freq_hz / c   # wave number

    # result is [num_beams, num_sidelobes]
    sin_sidelobes = sin_beam[:, np.newaxis] - 2 * q[np.newaxis, :] / k
    sidelobes = np.rad2deg(np.arcsin(sin_sidelobes))

    return sidelobes


def tx_range_from_rx_range(rx_distance, range_gates):
    """
    Calculates the transmitter-scatter straight-line distance of a bistatic geometry given
    the scatter->receiver distance and the total straight-line propagation distance.
    """
    tx_distance = 360. + 90. * range_gates - rx_distance

    return tx_distance


def azimuth_from_elevation(elevation, beam_dir):
    """
    Calculate the true azimuthal angle from a beam direction and elevation. All units are degrees.
    """
    return np.rad2deg(np.arcsin(np.sin(np.deg2rad(beam_dir)) / np.cos(np.deg2rad(elevation))))


def geolocate_scatter(tx_site, rx_site, elv, bmazm, slist, bmnum, gflg):
    """
    Geolocate the ground projection of scattered data from a SuperDARN radar.

    Parameters
    ----------
    tx_site: str
        Three-letter radar code, such as 'rkn' or 'sas'. Lowercase letters.
    rx_site: str
        Three-letter radar code, such as 'rkn' or 'sas', Lowercase letters.
    elv: np.array
        Array of shape [num_points] with the elevation data.
    bmazm: np.array
        Array of shape [num_points] with the zero-elevation beam azimuth directions relative to boresight.
    slist: np.array
        Array of shape [num_points] with the range-gate values for each data point.
    bmnum: np.array
        Array of shape [num_points] with the beam numbers for each data point.
    gflg: np.array
        Array of shape [num_points] with the ground scatter flag for each data point.

    Returns
    -------
    dict with keys:
        'scatter_location':     (lon, lat) position of scatter location for each point.
        'look_dir':             Direction CW of North for velocity vector of each point.
        'h_mid':                Virtual height of midpoint for each point.
        'h_rx':                 Virtual height seen by RX radar for each point.
        'h_tx':                 Virtual height seen by TX radar for each point.
        'rx_arc':               Geocentral angle travelled from Scatter-RX for each point.
        'tx_arc':               Geocentral angle travelled from TX-Scatter for each point.
        'total_arc':            Geocentral angle travelled from TX-Scatter-TX for each point.
        'rx_gnd_range':         Ground distance from Scatter-RX for each point.
        'tx_gnd_range':         Ground distance from TX-Scatter for each point.
        'gnd_range':            Total ground distance traversed from RX-Scatter-TX for each point.
        'alpha':                Scatter-RX-TX angle for each point
        'beam':                 Beam number for each point.
        'bmazm':                Beam azimuths relative to boresight for each point.
        'true_bmazm':           Beam azimuths relative to boresight for each point, correcting for elevation.
        'gsct':                 Groundscatter flag for each point.
        'group_range':          Group range for each point.
        'elv':                  Elevation values for each point.
        'slist':                Range gate values for each point.
        'velocity_correction':  Bistatic correction factor for velocity measurements.
    """
    r_e = 6378.0    # Radius of Earth, km
    geodesic = cartopy.geodesic.Geodesic()
    rx = sites[rx_site]
    tx = sites[tx_site]

    rx_tx_line = geodesic.inverse((rx['lon'], rx['lat']), (tx['lon'], tx['lat']))
    site_separation = rx_tx_line[0, 0] / 1000   # km
    site_arc = site_separation / r_e            # arc angle in radians
    cos_site_arc = np.cos(site_arc)
    sin_site_arc = np.sin(site_arc)
    rx_angle = rx_tx_line[0, 1]                 # angle CW of North of direct line at RX site, in degrees

    rad_elv = np.deg2rad(elv)
    true_az_dirs = rx['boresight'] + azimuth_from_elevation(elv, bmazm)  # CW of North

    alphas = np.abs(rx_angle - true_az_dirs)    # scatter-rx-tx angle on surface of Earth
    cos_alphas = np.cos(np.deg2rad(alphas))
    ranges = tx_range_from_rx_range(0, slist)   # LOS equivalent distance of signals
    r = np.float32(ranges)                      # Saved as low precision, so cast to higher precision here

    hv_mid = np.sqrt(r*r / 4 + r_e*r_e + r_e*r*np.sin(rad_elv)) - r_e   # virtual height at midpoint
    gamma = 2 * np.arcsin(r/2 * np.cos(rad_elv) / (r_e + hv_mid))       # total Earth arc angle

    numerator = 1 - cos_site_arc*np.cos(gamma) - sin_site_arc*np.sin(gamma)*cos_alphas
    denominator = cos_site_arc*np.sin(gamma) - sin_site_arc*np.cos(gamma)*cos_alphas

    tx_scatter_arc = np.arctan2(numerator, denominator)     # portion of gamma that is tx -> scatter
    rx_scatter_arc = gamma - tx_scatter_arc                 # portion of gamma that is scatter -> rx

    hv_rx = r_e * np.cos(rad_elv) / np.cos(rx_scatter_arc + rad_elv) - r_e      # virtual height of rx leg
    hv_tx = r_e * np.cos(rad_elv) / np.cos(tx_scatter_arc + rad_elv) - r_e      # virtual height of tx leg

    rx_scatter_dist = r_e * rx_scatter_arc                  # ground distance of rx leg
    tx_scatter_dist = r_e * tx_scatter_arc                  # ground distance of tx leg

    # Get scatter ground projection, and rx->scatter and tx->scatter directions at scatter location.
    rx_scatter_line = geodesic.direct((rx['lon'], rx['lat']), true_az_dirs, rx_scatter_dist * 1000)
    tx_scatter_line = geodesic.inverse((tx['lon'], tx['lat']), rx_scatter_line[:, :2])

    def normalize_angle(angle_deg):
        """Brings an angle (in degrees) into range [-180, 180)"""
        return ((angle_deg + 180) % 360) - 180

    angular_difference = normalize_angle(true_az_dirs - rx_tx_line[0, 1])

    # If boresight is CCW of rx-tx line (like RKN->INV pair), then forward points in a bistatic pair are points for
    # which the scatter point is CCW of the rx-tx line. Opposite for boresight CW of rx-tx line.
    if normalize_angle(rx['boresight'] - rx_tx_line[0, 1]) < 0:
        forward_points = angular_difference < 0
    else:
        forward_points = angular_difference > 0

    # Line which any velocity measurements lie along
    velocity_dirs = (rx_scatter_line[:, -1] + tx_scatter_line[:, -1]) / 2

    # Positive velocity means coming towards radar, so need to flip the direction for monostatic operations
    if tx_site == rx_site:
        velocity_dirs += 180
    # For bistatic radars, all positive velocities are towards the rx-tx line, so only flip direction for forward points
    else:
        velocity_dirs -= np.where(forward_points, 180, 0)   # positive

    # in bistatic setup, there is extra cos(beta/2) factor in doppler shift formula that we need to remove to
    # extract the true plasma velocity (or at least a better approximation)
    beta = np.arcsin(sin_site_arc * np.sin(np.deg2rad(alphas)) / np.sin(tx_scatter_arc))
    velocity_correction = 1 / np.cos(beta / 2)

    results = {'scatter_location': rx_scatter_line[:, :2],
               'look_dir': velocity_dirs,
               'h_mid': hv_mid,
               'h_rx': hv_rx,
               'h_tx': hv_tx,
               'rx_arc': rx_scatter_arc,
               'tx_arc': tx_scatter_arc,
               'total_arc': np.abs(rx_scatter_arc) + np.abs(tx_scatter_arc),
               'rx_gnd_range': rx_scatter_dist,
               'tx_gnd_range': tx_scatter_dist,
               'gnd_range': gamma * r_e,
               'alpha': alphas,
               'beam': np.ones(len(slist)) * bmnum,
               'bmazm': np.ones(len(slist)) * bmazm,
               'true_bmazm': azimuth_from_elevation(elv, bmazm),
               'gsct': gflg,
               'group_range': tx_range_from_rx_range(0, slist),
               'elv': elv,
               'slist': slist,
               'velocity_correction': velocity_correction
               }

    return results


# def triangulation(tx_site, rx_site, record):
#     """
#     Takes a list of FITACF records and runs geolocate_scatter on each record.
#
#     Parameters
#     ----------
#     tx_site: str
#         Three-letter radar code of transmitter site, such as 'rkn' or 'sas'. Lowercase letters.
#     rx_site: str
#         Three-letter radar code of receiver site, such as 'rkn' or 'sas'. Lowercase letters.
#     record: dict
#         FITACF record (dictionary) to process.
#
#     Returns
#     -------
#     List of geolocated scatter dictionaries, which have keys:
#         'scatter_location':     (lon, lat) position of scatter location for each point.
#         'look_dir':             Direction CW of North for velocity vector of each point.
#         'h_mid':                Virtual height of midpoint for each point.
#         'h_rx':                 Virtual height seen by RX radar for each point.
#         'h_tx':                 Virtual height seen by TX radar for each point.
#         'rx_arc':               Geocentral angle travelled from Scatter-RX for each point.
#         'tx_arc':               Geocentral angle travelled from TX-Scatter for each point.
#         'total_arc':            Geocentral angle travelled from TX-Scatter-TX for each point.
#         'rx_gnd_range':         Ground distance from Scatter-RX for each point.
#         'tx_gnd_range':         Ground distance from TX-Scatter for each point.
#         'gnd_range':            Total ground distance traversed from RX-Scatter-TX for each point.
#         'alpha':                Scatter-RX-TX angle for each point
#         'beam':                 Beam number for each point.
#         'bmazm':                Beam azimuths relative to boresight for each point.
#         'true_bmazm':           Beam azimuths relative to boresight for each point, correcting for elevation.
#         'gsct':                 Groundscatter flag for each point.
#         'group_range':          Group range for each point.
#         'elv':                  Elevation values for each point.
#         'slist':                Range gate values for each point.
#         'velocity_correction':  Bistatic correction factor for velocity measurements.
#     """
#     list_results = collections.defaultdict(list)    # For storing results of each record in scan
#
#     for rec in records:
#         try:
#             geolocated = geolocate_scatter(tx_site, rx_site, rec['elv'], rec['bmazm'], rec['slist'], rec['bmnum'],
#                                            rec['gflg'])
#             for k, v in geolocated.items():
#                 list_results[k].append(v)
#         except KeyError as err:
#             print('Warning: partial record', err)
#
#     results = {k: np.concatenate(v) for k, v in list_results.items()}
#
#     return results


def extract_values(param, results, record):
    """
    Extracts param from geolocation results and the record itself.

    Parameters
    ----------
    param: str
        Parameter to extract.
    results: dict
        Dictionary that is returned by geolocate_scatter() function.
    record: dict
        Dictionary of a FITACF record.

    Returns
    -------
    values: Any
        Whatever is held under 'param' in results or record dicts.
    """
    values = None
    if param in results.keys():
        values = results[param]
    else:   # anything not extracted in geolocation function
        try:
            values = record[param]
        except KeyError as err:
            print(f'Parameter {param} unrecognized: {err}')
    return values


def geolocate_record(record, rx_site, tx_site, min_hv: float = 100, extra_params=None):
    """
    Geolocates data in record, accounting for sidelobe points in a bistatic arrangement.

    Parameters
    ----------
    record: dict
        A single FITACF record
    rx_site: str
        Three-letter receiver radar code, such as 'rkn' or 'sas'. Lowercase letters
    tx_site: str
        Three-letter transmitter radar code, such as 'rkn' or 'sas'. Lowercase letters
    min_hv: float
        Minimum virtual height in km, used for identifying side lobe scatter. Default 100.
    extra_params: list[str]
        List of extra parameters to extract from the geolocation results or the record, such as 'p_l' or 'v'.

    Returns
    -------
    dict with keys:
        'scatter_location':     (lon, lat) position of scatter location for each point.
        'look_dir':             Direction CW of North for velocity vector of each point.
        'h_mid':                Virtual height of midpoint for each point.
        'h_rx':                 Virtual height seen by RX radar for each point.
        'h_tx':                 Virtual height seen by TX radar for each point.
        'rx_arc':               Geocentral angle travelled from Scatter-RX for each point.
        'tx_arc':               Geocentral angle travelled from TX-Scatter for each point.
        'total_arc':            Geocentral angle travelled from TX-Scatter-TX for each point.
        'rx_gnd_range':         Ground distance from Scatter-RX for each point.
        'tx_gnd_range':         Ground distance from TX-Scatter for each point.
        'gnd_range':            Total ground distance traversed from RX-Scatter-TX for each point.
        'alpha':                Scatter-RX-TX angle for each point
        'beam':                 Beam number for each point.
        'bmazm':                Beam azimuths relative to boresight for each point.
        'true_bmazm':           Beam azimuths relative to boresight for each point, correcting for elevation.
        'gsct':                 Groundscatter flag for each point.
        'group_range':          Group range for each point.
        'elv':                  Elevation values for each point.
        'slist':                Range gate values for each point.
        'velocity_correction':  Bistatic correction factor for velocity measurements.
        plus extra_params and their fields.
    """
    results = geolocate_scatter(tx_site, rx_site, record['elv'], record['bmazm'], record['slist'], record['bmnum'],
                                record['gflg'])

    groundscatter = extract_values('gsct', results, record)
    slist = extract_values('slist', results, record)

    if len(slist) == 0:     # No valid points found
        return

    geodesic = cartopy.geodesic.Geodesic()
    rx = sites[rx_site]
    tx = sites[tx_site]

    distances = geodesic.inverse((rx['lon'], rx['lat']), (tx['lon'], tx['lat']))
    site_separation = distances[0, 0] / 1000  # km
    min_range_gate = round((site_separation - 360) / 90.)               # based on great circle distance between sites

    impossible_points = slist < min_range_gate                                          # unphysical (faster than light)
    direct_points = np.logical_and(min_range_gate + 2 >= slist, groundscatter == 1)     # direct mode

    if rx_site == tx_site:
        valid_points = np.logical_and(~impossible_points, ~direct_points)   # neither impossible nor direct
    else:
        valid_points = np.full(slist.shape, True)                           # Assume all points are valid

    azms = extract_values('bmazm', results, record)                 # Azimuth of each point
    sidelobes = sidelobe_finder(azms, record['tfreq'] * 1000)       # Sidelobes for all points
    phi0 = extract_values('phi0', results, record)                  # main-intf array phase offsets
    lobe_num = np.zeros(slist.shape)                                # which lobe each point came from

    for lobe in range(sidelobes.shape[-1]):         # Loop through all sidelobes
        misplaced_points = results['rx_arc'] < 0.0      # all points that are being placed behind the radar
        misplaced_points |= results['h_rx'] < min_hv    # unphysically low
        misplaced_points |= results['h_tx'] < min_hv    # unphysically low
        if np.count_nonzero(misplaced_points) == 0:     # all points accounted for
            break

        lobe_num[misplaced_points] = lobe               # points attributed to this lobe

        # Geolocate points based on new assumed sidelobe location
        new_elv = find_elevation(rx_site, sidelobes[misplaced_points, lobe], record['tfreq']*1000,
                                 phi0[misplaced_points])
        bmnum = extract_values('beam', results, record)
        corrected_results = geolocate_scatter(tx_site, rx_site, new_elv, sidelobes[misplaced_points, lobe],
                                              slist[misplaced_points], bmnum[misplaced_points],
                                              groundscatter[misplaced_points])
        for k, v in results.items():
            results[k][misplaced_points] = corrected_results[k]     # update with new values

    # Any points that didn't fit a sidelobe should be wiped out
    misplaced_points = results['rx_arc'] < 0.0      # all points that are being placed behind the radar
    misplaced_points |= results['h_rx'] < min_hv    # unphysically low
    misplaced_points |= results['h_tx'] < min_hv    # unphysically low
    results['valid'] = np.logical_and(valid_points, ~misplaced_points)  # Keep track of "good" located points
    for k, v in results.items():
        if not np.issubdtype(v.dtype, np.integer):
            v[~valid_points] = np.nan  # I don't want to see these when plotting

    results['lobe'] = lobe_num

    # Get any other parameters from the data that are requested (easy to do it here where results and record are kept)
    if extra_params is not None:
        for param in extra_params:
            if param not in results.keys():
                results[param] = extract_values(param, results, record)
            if param == 'v':
                results[param] *= extract_values('velocity_correction', results, record)

    return results
