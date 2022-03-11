import os
import sklearn
import _pickle as cPickle
import gzip


def main(pth, scaler, test_split=0.15):
    """
    Args:
        pth: str, full path to folder where raw data are stored
        scaler: scikit learn object to define scaling
        test_split : double, default (0.15). between 0-1, determines size of test set
    Returns:

    """

    # create database from trial data
    # X is array (tensor) of shape (trial x channels x frames)
    # y is array of shape (labels)
    # z is array of subject names



    X, y, subs = zoo2database(pth)

    # split into train and test sets
    X_train, X_test, y_train, y_test, subs_train, subs_test = train_test_split.split(X, y, subs, test_split)

    # Signal amplitude scaling scaling
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


if __name__ == "__main__":
    pkl_zip_file = 'data_raw_2021.pkl.zip'
    with gzip.open(os.path.join(os.getcwd(), pkl_zip_file)) as fp:
        datapkl = cPickle.load(fp)

    # load sample data
    pth = os.path.join(os.getcwd(), 'sample_data')

    # set scaler to Standard (or choose others)
    scaler = sklearn.preprocessing.StandardScaler()

    # run main
    main(pth, scaler)
