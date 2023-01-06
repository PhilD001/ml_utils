import numpy as np
import scipy.stats as st
from scipy.fftpack import fft, fftfreq
from scipy.signal import argrelextrema
import operator

""" basic feature engineering, see : 
    https://github.com/jeandeducla/ML-Time-Series/blob/master/Neural_Network-Accelerometer-Features.ipynb
"""


def stat_area_features(x, te=1.0):
    mean_ts = np.mean(x, axis=1).reshape(-1, 1)  # mean
    max_ts = np.amax(x, axis=1).reshape(-1, 1)  # max
    min_ts = np.amin(x, axis=1).reshape(-1, 1)  # min
    std_ts = np.std(x, axis=1).reshape(-1, 1)  # std
    skew_ts = st.skew(x, axis=1).reshape(-1, 1)  # skew
    kurtosis_ts = st.kurtosis(x, axis=1).reshape(-1, 1)  # kurtosis
    iqr_ts = st.iqr(x, axis=1).reshape(-1, 1)  # interquartile rante
    mad_ts = np.median(np.sort(abs(x - np.median(x, axis=1).reshape(-1, 1)),
                               axis=1), axis=1).reshape(-1, 1)  # median absolute deviation
    area_ts = np.trapz(x, axis=1, dx=te).reshape(-1, 1)  # area under curve
    sq_area_ts = np.trapz(x ** 2, axis=1, dx=te).reshape(-1, 1)  # area under curve ** 2

    names = ['mean_ts', 'max_ts', 'min_ts', 'std_ts', 'skew_ts', 'kurtosis_ts', 'iqr_ts',  'mad_ts', 'area_ts',
             'sq_area_ts']

    return np.concatenate((mean_ts, max_ts, min_ts, std_ts, skew_ts, kurtosis_ts,
                           iqr_ts, mad_ts, area_ts, sq_area_ts), axis=1), names


def frequency_domain_features(x, te=1.0):
    # As the DFT coefficients and their corresponding frequencies are symetrical arrays
    # with respect to the middle of the array we need to know if the number of readings
    # in x is even or odd to then split the arrays...
    if x.shape[1] % 2 == 0:
        N = int(x.shape[1] / 2)
    else:
        N = int(x.shape[1] / 2) - 1
    xf = np.repeat(fftfreq(x.shape[1], d=te)[:N].reshape(1, -1), x.shape[0], axis=0)  # frequencies
    dft = np.abs(fft(x, axis=1))[:, :N]  # DFT coefficients

    # statistical and area features
    dft_features = stat_area_features(dft, te=1.0)
    # weighted mean frequency
    dft_weighted_mean_f = np.average(xf, axis=1, weights=dft).reshape(-1, 1)
    # 5 first DFT coefficients
    dft_first_coef = dft[:, :5]
    # 5 first local maxima of DFT coefficients and their corresponding frequencies
    dft_max_coef = np.zeros((x.shape[0], 5))
    dft_max_coef_f = np.zeros((x.shape[0], 5))
    for row in range(x.shape[0]):
        # finds all local maximas indexes
        extrema_ind = argrelextrema(dft[row, :], np.greater, axis=0)
        # makes a list of tuples (DFT_i, f_i) of all the local maxima
        # and keeps the 5 biggest...
        extrema_row = sorted([(dft[row, :][j], xf[row, j]) for j in extrema_ind[0]],
                             key=operator.itemgetter(0), reverse=True)[:5]
        for i, ext in enumerate(extrema_row):
            dft_max_coef[row, i] = ext[0]
            dft_max_coef_f[row, i] = ext[1]

    return np.concatenate((dft_features, dft_weighted_mean_f, dft_first_coef,
                           dft_max_coef, dft_max_coef_f), axis=1)


def make_feature_vector(x, channel_names, te=1.0):
    """ create basic features based on :
        https://github.com/jeandeducla/ML-Time-Series/blob/master/Neural_Network-Accelerometer-Features.ipynb

    Arguments:
        x : np.array(samples, time frames, channels)
        te : float. default=1.0. See original code, I don't know what this is

    Returns
        x : np.array(samples, features)

    Note:
        - original code updated (1) allow for processing of n channels instead of 3, (2) removal of channel correlation
          features
    """

    feats_all = []
    feats_all_names = []
    for ch in range(0, x.shape[2]):
        # Raw signals :  stat and area features
        features_t, feature_t_names = stat_area_features(x[:, :, ch], te=te)
        feature_t_names = [channel_names[ch] + x for x in feature_t_names]        # Jerk signals :  stat and area features
        # features_xt_jerk = stat_area_features((x[:, 1:, ch] - x[:, :-1, ch]) / te, te=te)

        # Raw signals : frequency domain features
        # features_f = frequency_domain_features(x[:, :, ch], te=1 / te)

        # Jerk signals : frequency domain features
        # features_xf_jerk = frequency_domain_features((x[:, 1:, ch] - x[:, :-1, ch]) / te, te=1 / te)

        # Raw signals correlation coefficient between axis
        # cor = np.empty((x.shape[0], 3))
        # for row in range(x.shape[0]):
        #     xyz_matrix = np.concatenate((x[row, :].reshape(1, -1), y[row, :].reshape(1, -1),
        #                                  z[row, :].reshape(1, -1)), axis=0)
        #     cor[row, 0] = np.corrcoef(xyz_matrix)[0, 1]
        #     cor[row, 1] = np.corrcoef(xyz_matrix)[0, 2]
        #     cor[row, 2] = np.corrcoef(xyz_matrix)[1, 2]

        # feats_ch = np.concatenate((features_t, features_f), axis=1)
        feats_all.append(features_t)
        feats_all_names.append(feature_t_names)

    # flatten lists and arrays
    feats_all = np.concatenate(feats_all, axis=1)
    feats_all_names = list(np.concatenate(feats_all_names).flat)

    return feats_all, feats_all_names
