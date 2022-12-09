import os
import numpy as np
import pandas as pd
from definitions import DIR_DATA


def load_database(db_name):
    """ provides access to database db_name """
    if db_name == 'keras_sample':
        X, y, user = load_keras_sample_time_series_db()
    elif db_name == 'HAR_sample':
        X, y, user = load_har_db()
    else:
        raise NotImplementedError('database {} not supporte'.format(db_name))
    return X, y, user


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
    users = None
    return X, y, users


def load_har_db():
    """" loads the UCL human activity recognition dataset
    see: https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
    """
    pth = os.path.join(DIR_DATA + os.sep)
    if len(os.listdir(pth)) == 0:
        raise IOError('please download and unzip the HAR dataset from '
                      'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip')

    # extract train and test set data
    X_train, y_train, X_test, y_test = _load_dataset_har(pth)

    # merge (splitting will occur later)
    X = np.vstack((X_train, X_test))
    y = np.vstack((y_train, y_test))

    # undo categorical
    y = np.argmax(y, axis=1)

    users = None
    return X, y, users


def _load_dataset_har(prefix=''):
    # load all train
    trainX, trainy = _load_dataset_group_har('train', prefix + 'HAR_sample' + os.sep)
    # load all test
    testX, testy = _load_dataset_group_har('test', prefix + 'HAR_sample' + os.sep)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    return trainX, trainy, testX, testy


def _load_dataset_group_har(group, prefix=''):
    filepath = prefix + group + os.sep + 'Inertial Signals' + os.sep
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = _load_group_har(filenames, filepath)
    # load class output
    y = _load_file_har(prefix + group + os.sep + 'y_' + group + '.txt')
    return X, y


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


def load_arrays(pth_save, flags):
    """loads arrays created by extract_arrays_from_dict"""
    try:
        X = np.load(os.path.join(pth_save, 'X_{}.npy'.format(flags)))
        y = np.load(os.path.join(pth_save, 'y_{}.npy'.format(flags)))
        c = np.load(os.path.join(pth_save, 'c_{}.npy'.format(flags)))
        s = np.load(os.path.join(pth_save, 's_{}.npy'.format(flags)))
    except FileNotFoundError:
        X = None
        y = None
        c = None
        s = None

    return X, y, c, s




