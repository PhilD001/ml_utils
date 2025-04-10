import os.path
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.fftpack import fft, fftfreq
from sklearn.feature_selection import VarianceThreshold
import hashlib

from definitions import DIR_DATA, DIR_RESULTS, DIR_ROOT
from utils.utils import load_args, save_args


def extract_features(x, channel_names, freq, feature_set, save_features=True, force_recompute=False, fsamp=1,
                     verbose=False):
    """ create basic features based on :
        https://github.com/jeandeducla/ML-Time-Series/blob/master/Neural_Network-Accelerometer-Features.ipynb

    Arguments:
        x :                 np.array(samples, time frames, channels)
        channel_names :     list. Names of channels
        freq :              int. Frequency of signal
        feature_set :       str. Choices {'test' or 'train'}. Suffix used to allow identification of saved features
        save_features :     flaot. default True. Save features to disk for quick reload after first run
        force_recompute :   Bool. default False. Forces recomputation of features even if previously saved to disk
        fsamp :             float. default=1.0. Used as constant for some features, see methods
        verbose :           Bool. Default=False. If true, messages are printed to screen

    Returns
        feats_all :         np.array(samples, features)
        feats_all_names :   list, names assocaited with each column of feats_all

    Note:
        - original code updated (1) allow for processing of n channels instead of 3, (2) removal of channel correlation
          features
    """
    arg_dict = load_args()

    if verbose:
        print('computing features for the {} set ...'.format(feature_set))

    if force_recompute:
        feats_all, feats_all_names = compute_features(x, channel_names, fsamp, freq, verbose)
        if save_features:
            features_pth = _create_features_unique_path(feature_set, x, arg_dict)
            _save_features(feats_all, feats_all_names, features_pth, verbose)
    else:
        features_pth = _create_features_unique_path(feature_set, x, arg_dict)
        if os.path.exists(features_pth):
            feats_all, feats_all_names = _load_saved_features(features_pth, verbose)
        else:
            feats_all, feats_all_names = compute_features(x, channel_names, fsamp, freq, verbose)
            if save_features:
                _save_features(feats_all, feats_all_names, features_pth, verbose)

    return feats_all, feats_all_names


def _create_features_unique_path(feat_set, x, arg_dict):
    """ use arguments to create a unique path in which to save processed features """
    # extract info that has an effect of content of feature set
    feat_dict = dict(database=arg_dict['database'],
                     channel_names=arg_dict['channel_names'],
                     train_test_split_by_user=arg_dict['train_test_split_by_user'],
                     test_ratio=arg_dict['test_ratio'],
                     validation_ratio=arg_dict['validation_ratio'],
                     segment_shape=arg_dict['segment_shape'],
                     n_classes=arg_dict['n_classes'])

    # extract relevant info that is only present is some datasets
    if 'classification' in arg_dict:
        feat_dict['classification'] = arg_dict['classification']
    if 'positions' in arg_dict:
        feat_dict['positions'] = arg_dict['positions']

    # extract information
    database = feat_dict['database']
    feat_str = str(feat_dict)
    feat_hex = string_to_hex(feat_str)
    d_shape = str(x.shape).replace(', ', '_')[1:-1]

    features_pth = os.path.join(DIR_DATA, database, 'features', feat_set + 'set_' + 'shape_' + d_shape + '_' + feat_hex)

    return features_pth


def string_to_hex(string):
    # Create a hash object
    hash_object = hashlib.md5(string.encode())

    # Get the hexadecimal representation of the hash
    hex_code = hash_object.hexdigest()

    return hex_code


def feature_clean(x_train, x_test, feature_names, verbose=False):
    """ basic clean regardless of feature selection true or false """

    # convert to pandas for easier operations
    df_X_train = pd.DataFrame(x_train, columns=feature_names)
    df_X_test = pd.DataFrame(x_test, columns=feature_names)

    # replace any duplicate names
    if any(df_X_train.columns.duplicated()):
        df_X_train = rename_duplicate_columns(df_X_train, verbose)
        df_X_test = rename_duplicate_columns(df_X_test, verbose)

    # Replace -inf and inf with nan
    df_X_train = df_X_train.replace([np.inf, -np.inf], np.nan)
    df_X_test = df_X_test.replace([np.inf, -np.inf], np.nan)

    # Check for NaN values in each column and return a boolean mask
    nan_mask_train = df_X_train.isna().any()
    nan_mask_test = df_X_test.isna().any()

    # Get the indices of columns with NaN values
    nan_column_train = nan_mask_train[nan_mask_train].index.tolist()
    nan_column_test = nan_mask_test[nan_mask_test].index.tolist()
    if len(nan_column_train) > 0:
        df_X_train = df_X_train.drop(columns=nan_column_train, errors='ignore')
        df_X_test = df_X_test.drop(columns=nan_column_train, errors='ignore')
    if len(nan_column_test) > 0:
        df_X_train = df_X_train.drop(columns=nan_column_test, errors='ignore')
        df_X_test = df_X_test.drop(columns=nan_column_test, errors='ignore')

    # convert back to nd arrays
    x_train = df_X_train.to_numpy()
    x_test = df_X_test.to_numpy()
    feature_names = list(df_X_train.columns)

    return x_train, x_test, feature_names


def rename_duplicate_columns(df, verbose=False):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            if verbose:
                print('renaming feature {}'.format(newitem))
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df


def compute_features(x, channel_names, te, freq, verbose=False):
    # check if X data are finite
    if not np.isfinite(x).any():
        raise ValueError('Non-finite data found')

    # intialize feature list
    feats_all = []
    feats_all_names = []

    # extract basic statistical features for all channels
    for ch in range(0, x.shape[2]):
        features_stats, feature_stats_names = stat_area_features(x[:, :, ch], te=te)
        feature_stats_names = [channel_names[ch] + x for x in feature_stats_names]
        feats_all.append(features_stats)
        feats_all_names.append(feature_stats_names)

    # extract basic frequency domain features for all channels
    for ch in range(0, x.shape[2]):
        features_freq, feature_freq_names = frequency_domain_features(x[:, :, ch], te=te)
        feature_freq_names = [channel_names[ch] + x for x in feature_freq_names]
        feats_all.append(features_freq)
        feats_all_names.append(feature_freq_names)

    # flatten lists and arrays
    feats_all = np.concatenate(feats_all, axis=1)
    feats_all_names = list(np.concatenate(feats_all_names).flat)

    return feats_all, feats_all_names


def stat_area_features(x, te=1):
    """ compute basic statistical features based on
         https://github.com/jeandeducla/ML-Time-Series/blob/master/Neural_Network-Accelerometer-Features.ipynb
    """

    # extract features
    mean_ts = np.mean(x, axis=1).reshape(-1, 1)  # mean
    max_ts = np.amax(x, axis=1).reshape(-1, 1)  # max
    min_ts = np.amin(x, axis=1).reshape(-1, 1)  # min
    std_ts = np.std(x, axis=1).reshape(-1, 1)  # std
    iqr_ts = st.iqr(x, axis=1).reshape(-1, 1)  # interquartile rante
    mad_ts = np.median(np.sort(abs(x - np.median(x, axis=1).reshape(-1, 1)), axis=1), axis=1).reshape(-1, 1)  # median absolute deviation
    area_ts = np.trapz(x, axis=1, dx=te).reshape(-1, 1)  # area under curve
    sq_area_ts = np.trapz(x ** 2, axis=1, dx=te).reshape(-1, 1)  # area under curve ** 2
    temp = np.quantile(x, (.25, .5, .75), axis=1).transpose()
    q25_ts, med_ts, q_75_ts = temp[:, 0].reshape(-1, 1), temp[:, 1].reshape(-1, 1), temp[:, 2].reshape(-1, 1)

    features = np.concatenate((mean_ts, max_ts, min_ts, std_ts, iqr_ts, mad_ts, area_ts, sq_area_ts, q25_ts, med_ts,
                               q_75_ts), axis=1)

    # feature names
    feature_names = ['mean_ts', 'max_ts', 'min_ts', 'std_ts', 'iqr_ts',  'mad_ts', 'area_ts', 'sq_area_ts', 'q25_ts',
                     'median_ts', 'q75_ts']

    return features, feature_names


def frequency_domain_features(x, te=1.0):
    """ compute basic statistical features based on frequency domain analysis
        https://github.com/jeandeducla/ML-Time-Series/blob/master/Neural_Network-Accelerometer-Features.ipynb
    """

    if x.shape[1] % 2 == 0:
        N = int(x.shape[1] / 2)
    else:
        N = int(x.shape[1] / 2) - 1
    xf = np.repeat(fftfreq(x.shape[1], d=te)[:N].reshape(1, -1), x.shape[0], axis=0)  # frequencies
    dft = np.abs(fft(x, axis=1))[:, :N]  # DFT coefficients

    # statistical and area features
    dft_features = stat_area_features(dft, te=te)
    feature_names = dft_features[1]

    # weighted mean frequency
    dft_weighted_mean_f = np.average(xf, axis=1, weights=dft).reshape(-1, 1)
    feature_names.append('dft_weighted_mean_f')

    # 5 first DFT coefficients
    dft_first_coef = dft[:, :5]
    feature_names.append('dft_coeff1')
    feature_names.append('dft_coeff2')
    feature_names.append('dft_coeff3')
    feature_names.append('dft_coeff4')
    feature_names.append('dft_coeff5')

    return np.hstack([dft_features[0], dft_weighted_mean_f, dft_first_coef]), feature_names


def get_flirt_acc_features(x, freq, window_length=None, window_step_size=None, verbose=False):
    """ extract acceleration features from x using flirt
    Arguments
        x                   ...  nd.array. trials x frames x 3
        freq                ...  float, frequency of signal
        window_length       ...  int, length of x to use for feature extraction. Default (None) : use entire signal
        window_step_size    ...  int, length of step size. Default (None) : a single step over entire signal
        verbose             ...  bool, default False. If True, print information to screen
    Returns
        features            ... np.array. trials x n features. All features computed by flirt for acceleration
        feature_names       ... list, length n features. Names associated with each feature
    """

    # check inputs
    if window_length is None:
        window_length = x.shape[1]
    if window_step_size is None:
        window_step_size = x.shape[1]

    # convert x to dataframe
    if x.shape[2] < 3:
        raise IOError('requires 3D acceleration input')
    if x.shape[2] > 3:
        if verbose:
            print('reducing number of channels to 3, by selecting first 3 channels, check that these are acc xyz')
        x = x[:, : 0, 3]

    # convert tensor for pandas df
    features_list = []
    for i in range(0, x.shape[0]):
        x_i = x[i, :, :]   # nd array rows x channel
        x_i_df = pd.DataFrame(x_i, columns=['acc_x', 'acc_y', 'acc_z'])
        df = flirt.get_acc_features(x_i_df, window_length=window_length, window_step_size=window_step_size,
                                    data_frequency=freq)
        features_list.append(df.to_numpy())
        if i == 0:
            feature_names = list(df.columns)
            n_digits = len(str(x.shape[0]))-1

    # convert list of numpy to numpy
    features = np.array(features_list).squeeze(axis=1)
    return features, feature_names


def get_flirt_ecg_features(x, freq, window_length=None, window_step_size=None, verbose=False):
    """ extract ecg (hrv) features from x using flirt
    Arguments
        x                   ...  nd.array. trials x frames x 1
        freq                ...  float, frequency of signal
        window_length       ...  int, length of x to use for feature extraction. Default (None) : use entire signal
        window_step_size    ...  int, length of step size. Default (None) : a single step over entire signal
        verbose             ...  bool, default False. If True, print information to screen
    Returns
        features            ... np.array. trials x n features. All features computed by flirt for acceleration
        feature_names       ... list, length n features. Names associated with each feature
    """

    # check inputs
    if window_length is None:
        window_length = x.shape[1]
    if window_step_size is None:
        window_step_size = x.shape[1]

    # convert tensor for pandas df
    features_list = []
    df = flirt.get_hrv_features(x, window_length=window_length, window_step_size=window_step_size)
    features_list.append(df.to_numpy())
    feature_names = list(df.columns)

    # convert list of numpy to numpy
    features = np.array(features_list).squeeze(axis=1)
    return features, feature_names


def _load_saved_features(features_pth, verbose=False):
    if verbose:
        rel_pth = os.path.relpath(features_pth, DIR_ROOT)
        print('loading previously computed features from: {}'.format(rel_pth))
    feats_all = np.load(os.path.join(features_pth, 'features.npy'))
    feats_all_names = np.load(os.path.join(features_pth, 'feature_names.npy'))
    return feats_all, feats_all_names


def _save_features(feats_all, feats_all_names, features_pth, verbose=False):
    """ save features to disk for quicker future runs"""
    if not os.path.exists(features_pth):
        os.makedirs(features_pth)

    # save features to disk
    features = os.path.join(features_pth, 'features.npy')
    feature_names = os.path.join(features_pth, 'feature_names.npy')
    np.save(features, feats_all)
    np.save(feature_names, feats_all_names)

    if verbose:
        rel_pth = os.path.relpath(features_pth, DIR_ROOT)
        print('features saved to: {}'.format(rel_pth))


def feature_selection(x_train, x_test, feature_names, correlation_threshold=0.95, variance_threshold=0.01,
                      verbose=False):


    """ some simple feature reduction algorithms
    see  https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/

    """
    if verbose:
        print('running feature reduction algorithms ...')
        print('original number of features = {}'.format(x_test.shape[1]))

    # convert to pandas for easier operations
    df_X_train = pd.DataFrame(x_train, columns=feature_names)
    df_X_test = pd.DataFrame(x_test, columns=feature_names)

    # remove features with 0 variance
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(df_X_train)
    constant_columns = [column for column in df_X_train.columns if
                        column not in df_X_train.columns[constant_filter.get_support()]]
    if len(constant_columns) > 0:
        if verbose:
            print('removing {} features with variance of 0'.format(len(constant_columns), variance_threshold))
        df_X_train.drop(labels=constant_columns, axis=1, inplace=True)
        df_X_test.drop(labels=constant_columns, axis=1, inplace=True)

    # remove features with low variance
    constant_filter = VarianceThreshold(threshold=variance_threshold)
    constant_filter.fit(df_X_train)
    constant_columns = [column for column in df_X_train.columns if
                        column not in df_X_train.columns[constant_filter.get_support()]]
    if len(constant_columns) > 0:
        if verbose:
            print('removing {} features with variance lower than {}'.format(len(constant_columns), variance_threshold))
        df_X_train.drop(labels=constant_columns, axis=1, inplace=True)
        df_X_test.drop(labels=constant_columns, axis=1, inplace=True)

    # remove duplicate features
    df_X_train_T = df_X_train.T
    unique_features = df_X_train_T.drop_duplicates(keep='first').T
    duplicated_features = [dup_col for dup_col in df_X_train.columns if dup_col not in unique_features.columns]
    if len(duplicated_features) > 1:
        if verbose:
            print('removing {} duplicate features'.format(len(duplicated_features)))
        df_X_train.drop(labels=duplicated_features, axis=1, inplace=True)
        df_X_test.drop(labels=duplicated_features, axis=1, inplace=True)

    # remove correlated features
    correlated_features = set()
    correlation_matrix = df_X_train.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    if len(correlated_features) > 0:
        if verbose:
            print('removing {} correlated features'.format(len(correlated_features)))
        df_X_train.drop(labels=correlated_features, axis=1, inplace=True)      # Drop features
        df_X_test.drop(labels=correlated_features, axis=1, inplace=True)      # Drop features

    # convert back to nd arrays
    x_train = df_X_train.to_numpy()
    x_test = df_X_test.to_numpy()
    feature_names = list(df_X_train.columns)

    if verbose:
        print('final number of features = {}'.format(x_test.shape[1]))

    return x_train, x_test, feature_names
