import numpy as np


def subject_wise_split(Participant,subject_wise,split=0.10,seed=42):
    '''
    Input:
    Participant = array of Classes / array[Y]
    Subject_wise = True or Flase. True = subject-wise split approch and Flase = Random Split
    split = float number between 0 to 1. Default value = 0.10. percentege spilt.
    seed = int. seed selector for numpy random number generator.

    Return:
    train_index = array[int64], indexes for training set.
    test_index = array[int64], indexes for test set.
    extract = array[string], participants extracted for test set.'''
    np.random.seed(seed)
    if subject_wise:
        UniqParti=np.unique(Participant)
        num=np.round(UniqParti.shape[0]*split).astype('int64')
        np.random.shuffle(UniqParti)
        extract=UniqParti[0:num]
        test_index=np.array([],dtype='int64')
        for j in extract:
            test_index=np.append(test_index,np.where(Participant==j)[0])
        train_index=np.delete(np.arange(len(Participant)),test_index)
        np.random.shuffle(test_index)
        np.random.shuffle(train_index)

    else:
        I=np.arange(len(Participant)).astype('int64')
        np.random.shuffle(I)
        num=np.round(Participant.shape[0]*split).astype('int64')
        test_index=I[0:num]
        train_index=I[num:]
        extract=np.unique(Participant[test_index])
    return train_index,test_index,extract
