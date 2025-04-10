import os
import numpy as np
import pandas as pd
from tensorflow import keras

from definitions import DIR_DATA


def load_database(arg_dict):
    """ provides access to database

    Note:
    - Regarding organization of data tensors, Chollet, Deep Learning with Python, section 2.2.10m recommends:
    "Whenever time matters in your data (or the notion of sequence order), it makes sense to store it in a 3D tensor with
    an explicit time axis. Each sample can be encoded as a sequence of vectors (a 2D tensor), and thus a batch of data
    will be encoded as a 3D tensor (see figure 2.3)."
    """
    conditions = None
    if arg_dict['database'] == 'keras_sample':
        X, y, user, channel_names, meta_data = load_keras_sample_time_series_db()
    elif arg_dict['database'] == 'HAR_sample':
        X, y, user, channel_names, meta_data = load_har_db(channel_names=arg_dict['channel_names'])
    elif arg_dict['database'] == 'Honda':
        X, y, user, channel_names, meta_data = load_honda_db()
    else:
        raise NotImplementedError('database {} not supported'.format(arg_dict['database']))

    # overwrite channel names if required
    arg_dict['channel_names'] = channel_names

    return X, y, user, channel_names, conditions, meta_data


def load_keras_sample_time_series_db():
    """ loads  keras_sample data set for testing models
    - see : https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    """

    # get data
    fl_train = os.path.join(DIR_DATA, 'keras_sample', 'FordA_TRAIN.tsv')
    fl_test = os.path.join(DIR_DATA, 'keras_sample', 'FordA_TEST.tsv')
    if os.path.exists(fl_train):
        print('keras_sample train data previously downloaded, loading from local drive at {}'.format(fl_train))
        print('keras_sample test data previously downloaded, loading from local drive at {}\n'.format(fl_test))
        root_dir = os.path.join(DIR_DATA, 'keras_sample') + os.path.sep
    else:
        raise NotImplementedError('problem automatically extracting data from internet')
        root_dir = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    fl_x = os.path.join(DIR_DATA, 'keras_sample', 'X.npy')
    fl_y = os.path.join(DIR_DATA, 'keras_sample', 'y.npy')
    if os.path.exists(fl_x):
        print('extracting data from previously saved numpy arrays at:')
        print('{}'.format(fl_x))
        print('{}\n'.format(fl_y))

        X = np.load(fl_x)
        y = np.load(fl_y)
    else:
        x_train, y_train = _readucr(root_dir + "FordA_TRAIN.tsv")
        x_test, y_test = _readucr(root_dir + "FordA_TEST.tsv")

        # merge (splitting will occur later)
        X = np.vstack((x_train, x_test))
        y = np.hstack((y_train, y_test))

        # turn into multivariate format (trials x frames x channels)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Now we shuffle the set because we will be using the validation_split option later when training.
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]

        # Standardize the labels to positive integers. The expected labels will then be 0 and 1.
        y[y == -1] = 0

        # save data to numpy
        print('saving keras_sample data as numpy arrays at {}\n'.format(DIR_DATA))
        np.save(os.path.join(DIR_DATA, 'keras_sample', 'X'), X)
        np.save(os.path.join(DIR_DATA, 'keras_sample', 'y'), y)

    # get users
    users = np.arange(0, len(y), 1)
    channel_names = ['motor_noise']
    meta_data = dict(freq=1)
    return X, y, users, channel_names, meta_data


def load_honda_db(sensor_type='IMU'):
    """ loads IMU + force pressure data from the Losig et al. public dataset. See paper here:

    interpolated IMU : https://drive.google.com/file/d/1vfNW-BHeErT7Sn6cJmFmE-sKbePAZGI8/view?usp=drive_link
    interpolated pressure: https://drive.google.com/file/d/1nPjoQFMUDvtco4Cmm9yf-NBQ2aR_kplP/view?usp=drive_link

    Returns:
        X: nd array (samples, frames, channels)
        y: list (samples)
        users: nd array (samples)
        channel_names: list. unique channel names in correct order
    """
    #todo add path to pressure

    if sensor_type == 'IMU':
        filepath = os.path.join(DIR_DATA, 'Honda_sample', 'interpolated_imu_sensor_df.csv')
    else:
        raise NotImplementedError

    df = pd.read_csv(filepath)
    x_df = df.drop(['time', 'sensor_location', 'walk_mode'], axis=1)
    y_df = df[['walk_mode', 'gait_cycle_id']]

    # Group by 'gait_cycle_id' and then remove the column (thanks ChatGPT)
    x = [group.drop(columns='gait_cycle_id').values for _, group in x_df.groupby('gait_cycle_id')]
    y_list = [group.drop(columns='gait_cycle_id').values for _, group in y_df.groupby('gait_cycle_id')]

    # create the y array
    y = [y[0, 0] for y in y_list]  # Extract the repeated string from each array

    # create the x tensor
    X = np.stack(x, axis=0)

    # todo: update users
    users = np.arange(1, len(y) + 1)

    # get channel names in order
    channel_names, indices = np.unique(df['sensor_location'], return_index=True)
    channel_names = list(channel_names[np.argsort(indices)])

    meta_data = None
    return X, y, users, channel_names, meta_data


def load_har_db(channel_names='all'):
    """" loads the UCL human activity recognition dataset
    see:
    sample loading code: https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
    database info: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    """
    pth = os.path.join(DIR_DATA + os.sep)
    if len(os.listdir(pth)) == 0:
        raise IOError('please download and unzip the HAR dataset from '
                      'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip')

    if channel_names[0] == 'all':
        channel_names = 'all'
    if channel_names == 'all':
        channel_names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                         'body_acc_x', 'body_acc_y', 'body_acc_z',
                         'body_gyro_x', 'body_gyro_y', 'body_gyro_z']

    # extract train and test set data
    X_train, y_train, X_test, y_test, user_train, user_test = _load_dataset_har(channel_names, pth)

    # merge (splitting will occur later)
    X = np.vstack((X_train, X_test))
    y = np.vstack((y_train, y_test))
    users = np.vstack((user_train, user_test))

    # undo categorical
    y = np.argmax(y, axis=1)

    # convert to class names (see activity_labels.txt)
    # labels are 1 less than in activity_labels file to account for 0 label
    y = [str(lbl) for lbl in y]
    y = ['walking' if item == '0' else item for item in y]
    y = ['walking_upstairs' if item == '1' else item for item in y]
    y = ['walking_downstairs' if item == '2' else item for item in y]
    y = ['sitting' if item == '3' else item for item in y]
    y = ['standing' if item == '4' else item for item in y]
    y = ['laying' if item == '5' else item for item in y]

    meta_data = dict(freq=50)
    return X, y, users, channel_names, meta_data


def _load_dataset_har(channel_names, prefix=''):
    # load all train
    trainX, trainy, trainuser = _load_dataset_group_har('train', channel_names, prefix + 'HAR_sample' + os.sep)
    # load all test
    testX, testy, testuser = _load_dataset_group_har('test', channel_names, prefix + 'HAR_sample' + os.sep)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = keras.utils.to_categorical(trainy)
    testy = keras.utils.to_categorical(testy)
    return trainX, trainy, testX, testy, trainuser, testuser


def _load_dataset_group_har(group, channel_names, prefix=''):
    filepath = prefix + group + os.sep + 'Inertial Signals' + os.sep
    # load all 9 files as a single array
    filenames = list()
    if 'total_acc_x' in channel_names:
        filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    if 'body_acc_x' in channel_names:
        filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    if 'body_gyro_x' in channel_names:
        filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = _load_group_har(filenames, filepath)
    # load class output
    y = _load_file_har(prefix + group + os.sep + 'y_' + group + '.txt')
    # load subject output
    user = _load_file_har(prefix + group + os.sep + 'subject_' + group + '.txt')
    return X, y, user


def _load_group_har(filenames, prefix=''):
    """ load a list of files into a 3D array of [samples, timesteps, features] """
    loaded = list()
    for name in filenames:
        data = _load_file_har(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded


def _load_file_har(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


def _readucr(filename):
    """ helper function to load text files """
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)





