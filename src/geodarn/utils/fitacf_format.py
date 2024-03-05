# This file is part of geodarn.
#
# geodarn is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# geodarn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with geodarn.
# If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2024 Remington Rohel

scan_specific_scalars = {
    'time.yr': int,
    'time.mo': int,
    'time.dy': int,
    'time.hr': int,
    'time.mt': int,
    'time.sc': int,
    'time.us': int,
    'intt.sc': int,
    'intt.us': int,
    'radar.revision.major': int,
    'radar.revision.minor': int,
    'origin.code': int,
    'origin.time': str,
    'origin.command': str,
    'cp': int,
    'stid': int,
    'txpow': int,
    'nave': int,
    'atten': int,
    'lagfr': int,
    'smsep': int,
    'ercod': int,
    'stat.agc': int,
    'stat.lopwr': int,
    'noise.search': float,
    'noise.mean': float,
    'channel': int,
    'scan': int,
    'offset': int,
    'rxrise': int,
    'txpl': int,
    'mpinc': int,
    'mppul': int,
    'mplgs': int,
    'mplgexs': int,
    'ifmode': int,
    'nrang': int,
    'frang': int,
    'rsep': int,
    'xcf': int,
    'tfreq': int,
    'mxpwr': int,
    'lvmax': int,
    'combf': str,
    'algorithm': str,
    'fitacf.revision.major': int,
    'fitacf.revision.minor': int,
    'noise.sky': float,
    'noise.lag0': float,
    'noise.vel': float,
    'tdiff': float
}

record_specific_scalars = {
    'bmnum': int,
    'bmazm': float
}

scan_specific_vectors = {
    'ptab': int,
    'ltab': int,
}

record_specific_vectors = {
    'pwr0': float,
    'slist': int,
    'nlag': int,
    'qflg': int,
    'gflg': int,
    'p_l': float,
    'p_l_e': float,
    'p_s': float,
    'p_s_e': float,
    'v': float,
    'v_e': float,
    'w_l': float,
    'w_l_e': float,
    'w_s': float,
    'w_s_e': float,
    'sd_l': float,
    'sd_s': float,
    'sd_phi': float,
    'x_qflg': int,
    'x_gflg': int,
    'x_p_l': float,
    'x_p_l_e': float,
    'x_p_s': float,
    'x_p_s_e': float,
    'x_v': float,
    'x_v_e': float,
    'x_w_l': float,
    'x_w_l_e': float,
    'x_w_s': float,
    'x_w_s_e': float,
    'phi0': float,
    'phi0_e': float,
    'elv': float,
    'elv_fitted': float,
    'elv_error': float,
    'elv_low': float,
    'elv_high': float,
    'x_sd_l': float,
    'x_sd_s': float,
    'x_sd_phi': float
}

