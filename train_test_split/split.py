import pandas as pd
import numpy as np


def subject_wise_split(x, y, participant, subject_wise=True, split=0.10, seed=42):
    """
    Arguments:
        x: nd.array, feature space
        y: nd.array, label class
        participant: nd.array, participant assiciated with each row in x and y
        subject_wise: bool, choices {True, False}. True = subject-wise split approach, False random-split
        split: float, number between 0 and 1. Default value = 0.10. percentage spilt for test set.
        seed: int. seed selector for numpy random number generator.

    Returns:
        x_train,y_train,x_test,y_test
        subject_train, subject_test = array[string], participants extracted for train and test set.
    """

    np.random.seed(seed)
    if subject_wise:
        uniq_parti = np.unique(participant)
        num = np.round(uniq_parti.shape[0]*split).astype('int64')
        np.random.shuffle(uniq_parti)
        extract = uniq_parti[0:num]
        test_index = np.array([], dtype='int64')
        for j in extract:
            test_index = np.append(test_index, np.where(participant == j)[0])
        train_index = np.delete(np.arange(len(participant)), test_index)
        np.random.shuffle(test_index)
        np.random.shuffle(train_index)

    else:
        index = np.arange(len(participant)).astype('int64')
        np.random.shuffle(index)
        num = np.round(participant.shape[0] * split).astype('int64')
        test_index = index[0:num]
        train_index = index[num:]

    x_train = x[train_index]
    y_train = y[train_index]
    x_test = x[test_index]
    y_test = y[test_index]
    subject_test = participant[test_index]
    subject_train = participant[train_index]

    return x_train, y_train, x_test, y_test, subject_train, subject_test


if __name__ == "__main__":
    data = pd.read_pickle('testdata.pkl')
    x = np.random.randint(0, 5, size=[data['Participant'].shape[0], 2])
    y = np.random.randint(0, 3, size=[data['Participant'].shape[0], 1])
    x_train, y_train, x_test, y_test, p_train, p_test = subject_wise_split(x, y, participant=data['Participant'],
                                                                           subject_wise=True, split=0.10, seed=42)
    print(np.unique(p_test))
    print(x_train.shape)
    print(x_test.shape)
