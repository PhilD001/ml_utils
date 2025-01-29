import numpy as np
import os
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras

from processing.processing import load_database
from machine_learning.models import build_model, tune_keras_model, set_keras_callbacks, tune_sklearn_model
from utils.utils import print_plot_save_results, evaluate_model, load_trained_model, save_args, subject_wise_split
from feature_engineering.features import extract_features, feature_selection
from definitions import DIR_RESULTS, RANDOM_STATE
from machine_learning.scaler import NoScalingScaler


def main(arg_dict):
    """ Main function to train, validate, and test sample ml algorithms.

    Arguments
        args_dict: dict, see __main__ for available options
    Returns
        None
    """
    # 0 - SET UP -------------------------------------------------------------------------------------------------------
    # set up root results directory (first run only)
    if not os.path.exists(DIR_RESULTS):
        os.mkdir(DIR_RESULTS)

    # 1 - LOAD DATA ----------------------------------------------------------------------------------------------------
    # - shape of data should be (samples, frames, channels), see Deep Learning with Python p 36
    start_time = time.time()
    X, y, user, ch_names, conditions, meta_data = load_database(arg_dict)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    label_map = {l: i for i, l in enumerate(le.classes_)}

    # 2 - TRAIN/TEST SPLIT DATA ----------------------------------------------------------------------------------------
    by_subject = arg_dict['train_test_split_by_user']
    test_size = arg_dict['test_ratio']
    X_train, X_test, y_train, y_test, users_train, users_test = subject_wise_split(X, y, user, subject_wise=by_subject,
                                                                                   test_size=test_size,
                                                                                   random_state=RANDOM_STATE)

    # 3 - SCALE DATA ---------------------------------------------------------------------------------------------------
    # -In the interest of preventing information about the distribution of the test set leaking into your model, you
    #  should fit the scaler on your training data only, then standardise both training and test sets with that scaler.
    # - tensor data (samples x channels x frames) must be reshaped for scaling tool
    if arg_dict['scale_signal']:
        sc = MinMaxScaler()
    else:
        sc = NoScalingScaler()
    X_train = sc.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = sc.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # add information to args_dict that will be used later
    arg_dict['segment_shape'] = X_test.shape[1::]
    arg_dict['n_classes'] = len(np.unique(y_test))
    save_args(arg_dict)

    # 4 - SELECT MODEL TYPE (FEATURE VS SIGNAL) ------------------------------------------------------------------------
    n_features = None
    if arg_dict['model_type'] == 'features':

        # 4 A: FEATURE-BASED MODEL -------------------------------------------------------------------------------------
        freq = meta_data['freq']
        X_test, feat_names = extract_features(X_test, ch_names, freq, feature_set='test',
                                              force_recompute=arg_dict['force_recompute_features'],
                                              verbose=arg_dict['verbose'])
        X_train, _ = extract_features(X_train, ch_names, freq, feature_set='train',
                                      force_recompute=arg_dict['force_recompute_features'], verbose=arg_dict['verbose'])

        # Feature selection
        if arg_dict['feature_selection']:
            X_train, X_test, feat_names = feature_selection(X_train, X_test, feat_names, verbose=arg_dict['verbose'])

        # save number of features
        n_features = len(feat_names)

        # Train model (or load pre-trained model)
        if arg_dict['trained_model_path']:
            model, arg_dict, history = load_trained_model(arg_dict)
        else:
            if arg_dict['tune']:
                # model, arg_dict = tune_sklearn_model(X_train, y_train, arg_dict)
                model, arg_dict = tune_sklearn_model(X_train, y_train, arg_dict, cv_folds=5, participants=users_train)

            else:
                model = build_model(arg_dict['model_name'])

            model.fit(X_train, y_train)
            history = None

    else:

        # 4 B: SIGNAL BASED MODEL --------------------------------------------------------------------------------------

        # One hot encode labels
        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

        # Train model (or load pre-trained model)
        input_shape = X.shape[1::]
        n_classes = len(np.unique(y_train, axis=0))

        if arg_dict['trained_model_path']:
            model, arg_dict, history = load_trained_model(arg_dict)
        else:
            if arg_dict['early_stop']:
                callback = set_keras_callbacks(early_stop_patience=arg_dict['patience'])
            else:
                callback = []

            if arg_dict['tune']:
                # initialize a tuner (here, RandomSearch). We use objective to specify the objective to select the best
                # models, and we use max_trials to specify the number of different models to try.
                # see : https://www.tensorflow.org/tutorials/keras/keras_tuner
                model, arg_dict = tune_keras_model(X_train, y_train, arg_dict, callback)
            else:
                units_input = arg_dict['n_filts_input']
                units_inner = arg_dict['n_filts_inner']
                units_output = arg_dict['n_filts_output']
                n_layers = arg_dict['n_layers']
                dropout = arg_dict['dropout']
                dropout_amt = arg_dict['dropout_amt']
                regularizer = arg_dict['regularizer']
                kernel_size = arg_dict['kernel_size']
                post_conv_layer = arg_dict['post_conv_layer']

                model = build_model(model_name=arg_dict['model_name'], n_classes=n_classes, input_shape=input_shape,
                                    units_input=units_input, units_inner=units_inner, units_output=units_output,
                                    n_layers=n_layers, dropout=dropout, dropout_amt=dropout_amt,
                                    regularizer=regularizer, kernel_size=kernel_size,  post_conv_layer=post_conv_layer)

            # train or retrain (for tuned model)
            history = model.fit(X_train, y_train, batch_size=arg_dict['batch_size'], epochs=arg_dict['epochs'],
                                validation_split=arg_dict['validation_ratio'], verbose=1, callbacks=callback)

    y_pred_train = model.predict(X_train)

    # 5 - EVALUATE TRAINED MODEL ---------------------------------------------------------------------------------------
    model_eval_dict_train, cm_train = evaluate_model(y_train, y_pred_train)

    # 6 - EVALUATE MODEL ON TEST SET -----------------------------------------------------------------------------------
    # - You should not tune your model to improve the score on the test set, see Chollet, Deep Learning with Python, p97
    model_eval_dict_test = {}
    cm_test = None
    if arg_dict['evaluate_on_test_set']:
        y_pred_test = model.predict(X_test)
        model_eval_dict_test, cm_test = evaluate_model(y_test, y_pred_test)

    # 7 - PRINT PLOT, SAVE RESULTS AND MODELS --------------------------------------------------------------------------
    elapsed_time = (time.time() - start_time)  # in hours
    run_time = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed_time))
    print_plot_save_results(arg_dict, model_eval_dict_train, model_eval_dict_test, cm_train, cm_test, history, y,
                            label_map, run_time, model, n_features)


if __name__ == '__main__':
    # todo create a settings file
    import argparse
    parser = argparse.ArgumentParser(description="Cough Detection Project")

    # Data set arguments
    parser.add_argument('--database', default='keras_sample', choices={'keras_sample', 'HAR_sample', 'Honda'})
    parser.add_argument('--channel_names', nargs='+', default='all')

    # General arguments
    parser.add_argument('--verbose', default=False, action='store_true', help='use flag to print information to screen')

    # arguments for choosing train/test split **************************************************************************
    parser.add_argument('--train_test_split_by_user', default=False, action='store_true')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--evaluate_on_test_set', default=False, action='store_true', help='after train')
    parser.add_argument('--trained_model_path', type=str, default=None, help='path to pre-trained model in results dir')

    # arguments for selection model type *******************************************************************************
    parser.add_argument('--model_name', nargs='+', default=['random_forest', 'svc'], help='see models.py for all choices')

    # arguments for data processing ************************************************************************************
    parser.add_argument('--scale_signal', default=False, action='store_true')

    # arguments for feature engineering ********************************************************************************
    parser.add_argument('--feature_selection', default=False, action='store_true')
    parser.add_argument('--save_features', default=True, action='store_true', help='save to disk for quick reload')
    parser.add_argument('--force_recompute_features', default=False, action='store_true', help='ignore save, recompute')

    # arguments for deep learning models only **************************************************************************
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_filts_input', type=int, default=64)
    parser.add_argument('--n_filts_inner', type=int, default=64)
    parser.add_argument('--n_filts_output', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--early_stop', default=False, action='store_true')
    parser.add_argument('--dropout', default=False, action='store_true')
    parser.add_argument('--dropout_amt', type=float, default=0.3, help='dropout ratio, must be [0,1]')
    parser.add_argument('--regularizer', type=str, default=None, choices=['L1', 'L2', None], help='keras regularizer')
    parser.add_argument('--regularizer_amt', type=float, default=0.01)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--post_conv_layer', type=str, default='batch_normalize', choices=['batch_normalize', 'max_pool'])
    parser.add_argument('--validation_ratio', type=int, default=0.2)

    # Model arguments
    parser.add_argument('--tune', default=False, action='store_true', help='automatic parameter tuning during training')
    parser.add_argument('--evaluation_metric', default='accuracy', help='metric for tuning')

    args = parser.parse_args()
    args_dict = vars(args)

    # Check model type : It will return True if any of the substrings in substring_list is contained in string.
    substring_list = ['cnn', 'lstm', 'imagenet', 'transformer']
    if any(substring in args_dict['model_name'] for substring in substring_list):
        args_dict['model_type'] = 'signal'
    else:
        args_dict['model_type'] = 'features'

    main(args_dict)
