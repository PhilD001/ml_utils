from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from tensorflow import keras
# from keras import Sequential, callbacks, optimizers, regularizers
# from tensorflow.keras.layers import Conv1D, Dense, BatchNormalization, GlobalAveragePooling1D, LSTM, Dropout


def build_model(model_name, grid_results=None, n_classes=None, input_shape=None, units=None, n_layers=None,
                dropout=None, regularizer=None, dropout_amt=0, regularizer_amt=0.001):

    if 'cnn' in model_name:
        model = cnn(units=units, dropout=dropout, n_layers=n_layers, n_classes=n_classes, input_shape=input_shape,
                    dropout_amt=dropout_amt, kernel_regularizer=regularizer,)
    elif 'lstm' in model_name:
        model = lstm(units=units, dropout=dropout, n_classes=n_classes, input_shape=input_shape,
                     dropout_amt=dropout_amt, kernel_regularizer=regularizer)
    elif 'svc' in model_name:
        model = svc(grid_results)
    elif 'random_forest' in model_name:
        raise NotImplementedError
        #model = random_forest(grid_results)
    elif 'xgboost' in model_name:
        raise NotImplementedError
        # model = xgboost(grid_results)
    else:
        raise NotImplementedError

    return model


def build_model_tune(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    dropout = hp.Boolean("dropout")
    dropout_amt = hp.Float("dropout_amt", min_value=0.1, max_value=0.8,step=0.1, sampling='linear')
    n_layers = hp.Int("n_layers", min_value=1, max_value=10, step=1)
    kernel_size = hp.Int("kernel_size", min_value=1, max_value=4, step=1)

    # call existing model-building code with the hyperparameter values.
    model = cnn(units=units, activation=activation, lr=lr, dropout=dropout, n_layers=n_layers, kernel_size=kernel_size,
                dropout_amt=dropout_amt)
    return model


def train_tune_feature_model(args_dict, X, y):

    # choose model
    if 'decision_tree' in args_dict['model_name']:
        model = DecisionTreeClassifier(random_state=0)
        hyper_params = dict()
        hyper_params['criterion'] = ['gini', 'entropy']
        hyper_params['min_samples_split'] = [2, 4, 6, 8, 10]
        hyper_params['max_depth'] = [2, 4, 6, 8, 10]

    elif 'svc' in args_dict['model_name']:
        model = SVC(random_state=0)
        hyper_params = dict()
        hyper_params['kernel'] = ['rbf', 'sigmoid']
        hyper_params['gamma'] = [1, 0.1, 0.01, 0.001]
        hyper_params['C'] = [1, 2, 3, 4, 5, 10]
    else:
        raise NotImplementedError

    # grid search
    print('tuning {} model based on following hyper parameters:'.format(args_dict['model_name']))
    print(hyper_params)
    search = GridSearchCV(model, hyper_params, scoring='accuracy')
    models = search.fit(X, y)
    return models


def random_forest(grid_results=None):
    """ build a ra"""

    # define search space
    if grid_results is None:
        hyper_params = dict()
        hyper_params['criterion'] = ['gini', 'entropy']
        hyper_params['min_samples_split'] = [2, 4, 6, 8, 10]
        hyper_params['max_depth'] = [2, 4, 6, 8, 10]
        # define model
        model = RandomForestClassifier(random_state=0)

        # define search
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)
        model = GridSearchCV(estimator=model, param_grid=hyper_params, cv=cv, scoring='accuracy')
    else:
        model = RandomForestClassifier(random_state=0,
                                       criterion=grid_results.best_params_['criterion'],
                                       min_samples_split=grid_results.best_params_['min_samples_split'],
                                       max_depth=grid_results.best_params_['max_depth'],
                                       )

    return model


def svc(grid_results=None, cross_val=False):
    """ build a SVC model. If no arguments, hyperparameters are tuned. Else, model is built with grid_result
    hyperparamter values
    """
    # define search space
    if grid_results is None:
        hyper_params = dict()
        hyper_params['kernel'] = ['rbf', 'sigmoid', 'linear']
        hyper_params['gamma'] = [10, 1, 0.1, 0.01, 0.001]   # The higher the gamma value it tries to exactly fit the training data set
        hyper_params['C'] = [0.1, 1, 2, 3, 4, 5, 10, 100]      # is the penalty parameter of the error term
        # define model
        model = SVC(random_state=0)

        # define search
        if cross_val:
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
            model = GridSearchCV(estimator=model, param_grid=hyper_params, cv=cv, scoring='accuracy')
        else:
            model = GridSearchCV(estimator=model, param_grid=hyper_params, scoring='accuracy')

    else:
        model = SVC(random_state=0,
                    kernel=grid_results.best_params_['kernel'],
                    gamma=grid_results.best_params_['gamma'],
                    C=grid_results.best_params_['C'],
                    )

    return model


def lstm(units=100, activation='relu', lr=0.001, dropout=False, n_classes=3, input_shape=(3840, 5),
         dropout_amt=0.2, kernel_regularizer=None):

    """ From https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

    Note:
        - Weight regularization can be applied to the bias connection within the LSTM nodes.
    """

    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    if dropout:
        model.add(Dropout(dropout_amt))
    model.add(Dense(units, activation=activation))
    model.add(Dense(n_classes, activation='softmax'))

    # dense layer to output classes
    opt = optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.add(Dense(n_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    return model


def cnn(units=64, activation='relu', lr=0.001, dropout=False, n_layers=3, n_classes=3, input_shape=(5, 3840),
        kernel_size=3, dropout_amt=0.2, kernel_regularizer=None, kernel_regularizer_amt=0.01):
    """a keras CNN ample time series model rewritten from keras
       see: https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
       """

    # set regularizers
    if kernel_regularizer == 'L1':
        kernel_regularizer = keras.regularizers.L1(kernel_regularizer_amt)
    elif kernel_regularizer == 'L2':
        kernel_regularizer = keras.regularizers.L2(kernel_regularizer_amt)

    # initialize model
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=units, kernel_size=kernel_size, input_shape=input_shape,
                                  padding='same', activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.BatchNormalization())

    # keep adding convolutional inner layers
    for n_layer in range(0, n_layers):
        model.add(keras.layers.Conv1D(filters=units, kernel_size=kernel_size, padding='same', activation=activation,
                                      kernel_regularizer=kernel_regularizer))
        model.add(keras.layers.BatchNormalization())

    # add an option for dropout
    if dropout:
        model.add(keras.layers.Dropout(dropout_amt))

    # finish with global average pooling
    model.add(keras.layers.GlobalAveragePooling1D())

    # dense layer to output classes
    opt = keras.optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.add(keras.layers.Dense(n_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    else:
        model.add(keras.layers.Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    return model


def set_keras_callbacks(early_stop_patience=50):
    """ a callback taken from a Keras example,
    see : https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    """
    callback = [keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stop_patience, verbose=1)]
    return callback
