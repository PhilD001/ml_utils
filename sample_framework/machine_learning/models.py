import os
import numpy as np
import shutil
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
import keras_tuner as kt

import sys

# Dynamically locate data_folder
current_dir = os.path.dirname(__file__)  # Get the directory of utils.py
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # Navigate to project root
definitions_folder = os.path.join(parent_dir, "definitions")
utils_folder = os.path.join(parent_dir, "utils_folder")

# Add data_folder to sys.path
sys.path.append(definitions_folder)
sys.path.append(utils_folder)

# Import the method from definitions
from definitions import DIR_RESULTS, RANDOM_STATE
from utils.utils import load_args


def build_model(model_name, grid_results=None, n_classes=None, input_shape=None, units_input=None, units_inner=None,
                units_output=None, n_layers=None, dropout=None, regularizer=None, dropout_amt=0, kernel_size=3,
                post_conv_layer='batch_normalize'):

    if 'cnn' in model_name:
        model = cnn(units_input=units_input, units_inner=units_inner, units_output=units_output, dropout=dropout,
                    n_layers=n_layers, n_classes=n_classes, input_shape=input_shape, kernel_size=kernel_size,
                    dropout_amt=dropout_amt, kernel_regularizer=regularizer, post_conv_layer=post_conv_layer)
    elif 'lstm' in model_name:
        model = lstm(units=units_input, dropout=dropout, n_classes=n_classes, input_shape=input_shape,
                     dropout_amt=dropout_amt, kernel_regularizer=regularizer)
    elif 'transformer' in model_name:
        model = transformer(input_shape=input_shape, n_classes=n_classes, dropout_amt=dropout_amt)
    elif 'svc' in model_name:
        model = svc(grid_results)
    elif 'random_forest' in model_name:
       model = random_forest(grid_results)
    elif 'gradient_boosting' in model_name:
        model = gradient_boosting(grid_results)
    else:
        raise NotImplementedError

    return model


def tune_keras_model(X_train, y_train, arg_dict, callback):
    """ see tutorial here:
    https://neptune.ai/blog/keras-tuner-tuning-hyperparameters-deep-learning-model
    """

    # set evaluation metric for tuning
    # todo : implement tuner by custom metric

    tuner = kt.Hyperband(hypermodel=build_keras_model_tune,  # tuning options set in build_model_tune function
                         objective=kt.Objective('accuracy', direction='max'),
                         max_epochs=int(np.ceil(arg_dict['epochs']*1.1)),  # set higher than expected epochs to converg
                         hyperband_iterations=1,                           # set as high as you can afford!
                         overwrite=True,
                         factor=3,
                         directory=DIR_RESULTS,
                         project_name='keras_tuner_results_temp',
                         )

    # Define the OracleCallback to display current objective metric value
    # oracle_callback = keras.callbacks.TensorBoard("/tmp/tb_logs")

    # start the search (always with early stop)
    tuner.search(X_train, y_train,
                 validation_split=arg_dict['validation_ratio'],
                 epochs=arg_dict['epochs'],
                 callbacks=[callback],
                 )

    # Get the optimal hyperparameters and build an untrained model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    # summary of search results including the hyperparameter values and evaluation results for each trial
    tuner.results_summary()

    return model, arg_dict


def tune_sklearn_model(X_train, y_train, arg_dict, cv_folds=5):
    """ tunes sklearn models using keras tuner"""

    # clean up from previous run
    if os.path.exists(os.path.join(DIR_RESULTS, 'sklearn_tuner_results_temp')):
        shutil.rmtree(os.path.join(DIR_RESULTS, 'sklearn_tuner_results_temp'))
    if os.path.exists('best_model.h5'):
        os.remove('best_model.h5')

    eval_metric = arg_dict['evaluation_metric']
    if eval_metric == 'accuracy':
        metric = metrics.accuracy_score
    elif eval_metric == 'roc_auc_score':
        metric = metrics.roc_auc_score
    else:
        raise NotImplementedError

    # tune
    tuner = kt.tuners.SklearnTuner(
        oracle=kt.oracles.BayesianOptimizationOracle(     # any tuner except HyperBand which is for neural networks
            objective=kt.Objective('score', 'max'),       # always set objective to Objective('score', 'max')
            max_trials=10),
        hypermodel=build_sklearn_model_tune,
        scoring=metrics.make_scorer(metric),
        cv=StratifiedKFold(cv_folds),
        overwrite=True,
        directory=DIR_RESULTS,
        project_name='sklearn_tuner_results_temp',
    )

    tuner.search(X_train, y_train)
    model = tuner.get_best_models(num_models=1)[0]
    arg_dict['model_name'] = str(type(model).__name__)

    return model, arg_dict


def build_sklearn_model_tune(hp):
    """ see https://keras.io/api/keras_tuner/tuners/sklearn/"""

    # load model names
    arg_dict = load_args()
    model_type = hp.Choice('model_type', arg_dict['model_name'])

    if model_type == 'random_forest':
        min_samp_leaf = hp.Int('min_samples_leaf ', min_value=1, max_value=10, step=2)
        max_feats = hp.Choice('max_features', values=['auto', 'sqrt', 'log2'])
        max_depth = hp.Int('max_depth', min_value=1, max_value=20, step=1)
        n = hp.Int('n_estimators', min_value=10, max_value=50, step=2)
        model = RandomForestClassifier(n_estimators=n, max_features=max_feats, max_depth=max_depth,
                                       min_samples_leaf=min_samp_leaf,
                                       random_state=RANDOM_STATE)

    elif model_type == 'svc':
        kernel = hp.Choice('kernel', values=['linear', 'rbf', 'poly'])
        gamma = hp.Float('gamma', min_value=1e-3, max_value=1, sampling="log")
        model = SVC(kernel=kernel, gamma=gamma)
    elif model_type == 'gradient_boosting':
        max_depth = hp.Int("max_depth", min_value=1, max_value=10, step=1)
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        n_estimators = hp.Int("n_estimators", min_value=10, max_value=200, step=10)
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr)
    elif model_type == 'MLP':
        solver = hp.Choice('solver', values=['lbfgs', 'sgd', 'adam'])
        hidden_layers = hp.Int('hidden_layers', min_value=2, max_value=10, step=2)
        activation = hp.Choice('activation', values=['identity', 'logistic', 'tanh', 'relu'])
        lr = hp.Choice('learning_rate', values=['constant', 'adaptive'])
        model = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layers, activation=activation, learning_rate=lr)
    else:
        raise NotImplementedError('Model type {} not implemented'.format(model_type))
    return model


def build_keras_model_tune(hp):
    units_input = hp.Int("units_input", min_value=32, max_value=128, step=32)
    units_inner = hp.Int("units_inner", min_value=32, max_value=128, step=32)
    units_output = hp.Int("units_output", min_value=32, max_value=128, step=32)
    dropout = hp.Boolean("dropout")
    dropout_amt = hp.Float("dropout_amt", min_value=0.1, max_value=0.3, step=0.1)
    n_layers = hp.Int("n_inner_layers", min_value=6, max_value=10, step=1)
    post_conv_layer = 'batch_normalize'
    kernel_size = hp.Int("kernel_size", min_value=2, max_value=4, step=1)
    kernel_regularizer = hp.Choice('kernel_regularizer', values=['L1', 'None'])
    activation = 'relu'
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    # call existing model-building code with the hyperparameter values.
    model = cnn(units_input=units_input, units_inner=units_inner, units_output=units_output, dropout=dropout,
                n_layers=n_layers, kernel_size=kernel_size, dropout_amt=dropout_amt, activation=activation,
                kernel_regularizer=kernel_regularizer, post_conv_layer=post_conv_layer, lr=lr)

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
    """ build a basrandom forest classifier"""
    if grid_results:
        raise NotImplementedError
    else:
        model = RandomForestClassifier()
    return model


def gradient_boosting(grid_results=None):
    """ build a basic gradient boosting classifier"""
    if grid_results:
        raise NotImplementedError
    else:
        model = GradientBoostingClassifier()
    return model


def svc(grid_results=None):
    """ build an SVC model """
    # define search space
    if grid_results:
        model = SVC(random_state=0, kernel=grid_results.best_params_['kernel'], gamma=grid_results.best_params_['gamma'],
                    C=grid_results.best_params_['C'])
    else:
        model = SVC()
    return model


def lstm(units=100, activation='relu', lr=0.001, dropout=False, n_classes=3, input_shape=(3840, 5),
         dropout_amt=0.2, kernel_regularizer=None):

    """ From https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

    Note:
        - Weight regularization can be applied to the bias connection within the LSTM nodes.
    """
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(units, return_sequences=False))
    if dropout:
        model.add(keras.layers.Dropout(dropout_amt))
    model.add(keras.layers.Dense(units, activation=activation))
    model.add(keras.layers.Dense(n_classes, activation='softmax'))

    # dense layer to output classes
    opt = keras.optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.add(keras.layers.Dense(n_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    else:
        model.add(keras.layers.Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    return model


def cnn(units_input=64, units_inner=64, units_output=64, activation='relu', lr=0.001, dropout=False, n_layers=3,
        n_classes=None, input_shape=None, kernel_size=3, dropout_amt=0.2, kernel_regularizer=None,
        kernel_regularizer_amt=0.01, post_conv_layer='batch_normalize'):
    """a keras CNN ample time series model rewritten from keras
       see: https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    """

    # get shape of data
    arg_dict = load_args()
    if input_shape is None:
        input_shape = tuple(arg_dict['segment_shape'])
    if n_classes is None:
        n_classes = arg_dict['n_classes']

    # set regularizers
    if kernel_regularizer == 'L1':
        kernel_regularizer = keras.regularizers.L1(kernel_regularizer_amt)
    elif kernel_regularizer == 'L2':
        kernel_regularizer = keras.regularizers.L2(kernel_regularizer_amt)
    elif kernel_regularizer == 'None':
        kernel_regularizer = None

    # initialize model
    model = keras.Sequential()

    # First convolution layer
    model.add(keras.layers.Conv1D(filters=units_input, kernel_size=kernel_size, input_shape=input_shape,
                                  padding='same', activation=activation, kernel_regularizer=kernel_regularizer))
    if post_conv_layer == 'batch_normalize':
        model.add(keras.layers.BatchNormalization())
    elif post_conv_layer == 'max_pool':
        model.add(keras.layers.MaxPooling1D())
    else:
        raise NotImplementedError

    # inner convolution layers (variable n)
    for n_layer in range(0, n_layers):
        model.add(keras.layers.Conv1D(filters=units_inner, kernel_size=kernel_size, padding='same', activation=activation,
                                      kernel_regularizer=kernel_regularizer))
        if post_conv_layer == 'batch_normalize':
            model.add(keras.layers.BatchNormalization())
        elif post_conv_layer == 'max_pool':
            model.add(keras.layers.MaxPooling1D())
        else:
            raise NotImplementedError

    # final convolution layer
    model.add(keras.layers.Conv1D(filters=units_output, kernel_size=kernel_size, padding='same', activation=activation,
                                  kernel_regularizer=kernel_regularizer))
    if post_conv_layer == 'batch_normalize':
        model.add(keras.layers.BatchNormalization())
    elif post_conv_layer == 'max_pool':
        model.add(keras.layers.MaxPooling1D())
    else:
        raise NotImplementedError

    # add an option for dropout
    if dropout:
        model.add(keras.layers.Dropout(dropout_amt))

    # finish with global average pooling
    model.add(keras.layers.GlobalAveragePooling1D())

    # dense layer to output classes
    opt = keras.optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.add(keras.layers.Dense(n_classes, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy',
        #               metrics=['accuracy', tf.keras.metrics.Mean(name='roc_auc', dtype=None)],
        #               optimizer=opt)
        model.compile(loss='binary_crossentropy',
                      metrics=['accuracy'], optimizer=opt)
    else:
        model.add(keras.layers.Dense(n_classes, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', metrics=['accuracy',
        #               tf.keras.metrics.Mean(name='roc_auc', dtype=None)],
        #               optimizer=opt)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    return model


def transformer(input_shape, lr=0.001, dropout_amt=0.2, n_classes=None, embed_dim=32, num_heads=2, ff_dim=64, num_transformer_blocks=2):
    """
       Builds and compiles a Transformer-based model for time-series classification.

       Parameters:
           input_shape (tuple): Shape of the input data (time_steps, features).
           embed_dim (int): Embedding dimension for attention layers.
           num_heads (int): Number of attention heads.
           ff_dim (int): Feedforward network dimension.
           num_transformer_blocks (int): Number of Transformer blocks.
           dropout_amt (float): Dropout rate for regularization.
           learning_rate (float): Learning rate for the optimizer.

       Returns:
           keras.Model: A compiled Transformer-based model.
       """
    inputs = keras.layers.Input(shape=input_shape)

    # Embedding: Convert the input into a higher-dimensional space
    x = keras.layers.Dense(embed_dim)(inputs)

    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Multi-Head Self-Attention
        attention_output = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        attention_output = keras.layers.Dropout(dropout_amt)(attention_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Feedforward Network
        ffn_output = keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = keras.layers.Dense(embed_dim)(ffn_output)
        ffn_output = keras.layers.Dropout(dropout_amt)(ffn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global pooling and output layer
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout_amt)(x)
    outputs = keras.layers.Dense(n_classes, activation="sigmoid")(x)

    # Build and compile the model
    model = keras.Model(inputs, outputs)

    opt = keras.optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.compile(loss='binary_crossentropy',
                      metrics=['accuracy'], optimizer=opt)
    else:
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
