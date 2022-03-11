import pandas as pd
import numpy as np


def subject_wise_split(x,y,participant, subject_wise=True, split=0.10, seed=42):
    """
    Input:
    x = featuers space
    y = class or output
    participant = array of Classes / array[Y]
    Subject_wise = True or False. True = subject-wise split approach and False = Random Split
    split = float number between 0 and 1. Default value = 0.10. percentage spilt for test set.
    seed = int. seed selector for numpy random number generator.

    Return:
    x_train,y_train,x_test,y_test
    extract = array[string], participants extracted for test set.
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
        I = np.arange(len(participant)).astype('int64')
        np.random.shuffle(I)
        num = np.round(participant.shape[0] * split).astype('int64')
        test_index = I[0:num]
        train_index = I[num:]
        extract = np.unique(participant[test_index])

    x_train=x[train_index]
    y_train=y[train_index]
    x_test=x[test_index]
    y_test=y[test_index]
    return x_train,y_train,x_test,y_test,extract


if __name__ == "__main__":
    data = pd.read_pickle('testdata.pkl')
    x=np.random.randint(0,5,size=[data['Participant'].shape[0],2])
    y=np.random.randint(0,3,size=[data['Participant'].shape[0],1])
    x_train,y_train,x_test,y_test,extract = split.subject_wise_split(x,y,participant=data['Participant'],subject_wise=True, split=0.10, seed=42)
    print(extract)
    print(x_train.shape)
    print(x_test.shape)
