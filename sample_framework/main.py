import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras

from processing.processing import load_database
from machine_learning.models import build_model, build_model_tune, set_keras_callbacks
from utils.utils import print_plot_save_results
from feature_engineering.features import make_feature_vector


# todo: look at LSTM CNN models
# https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification


def main(arg_dict):
    """ Main function to train, validate, and test sample ml algorithms.

    Arguments
        args_dict: dict, see __main__ for available options
    Returns
        None
    """

    # LOAD DATA --------------------------------------------------------------------------------------------------------
    # - shape of data should be (samples, frames, channels), see Deep Learning with Python p 36
    start_time = time.time()
    X, y, user, channel_names = load_database(arg_dict['data'], arg_dict['channels'])

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    label_map = {l: i for i, l in enumerate(le.classes_)}

    # TRAIN/TEST SPLIT DATA --------------------------------------------------------------------------------------------
    # # todo: add choice to do subject wise split
    if arg_dict['train_test_split_by_user']:
        raise NotImplementedError
    else:
        indices = np.arange(len(y))
        X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indices,
                                                                                     test_size=arg_dict['test_ratio'],
                                                                                     random_state=42)

    # SCALE DATA -------------------------------------------------------------------------------------------------------
    # -In the interest of preventing information about the distribution of the test set leaking into your model, you
    #  should fit the scaler on your training data only, then standardise both training and test sets with that scaler.
    # - tensor data (samples x channels x frames) must be reshaped for scaling tool
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = sc.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # ADD SOME INFO TO ARGS_DICT ---------------------------------------------------------------------------------------
    args_dict['segment_shape'] = X_test.shape[1::]
    args_dict['n classes'] = len(np.unique(y_test))

    # SELECT MODEL TYPE ------------------------------------------------------------------------------------------------
    if arg_dict['model_type'] == 'features':
        # only works for acceleration for now
        print("\ncomputing features...".format(X_train.shape[1]))

        X_train, feature_names = make_feature_vector(X_train, channel_names, te=1 / 50)
        X_test, _ = make_feature_vector(X_test, channel_names, te=1 / 50)

        print("computed {} features".format(X_train.shape[1]))

        grid_search = build_model(model_name=arg_dict['model_name'])

        # execute search for hyperparameters
        print('executing search for best hyperparameters on the training set...')
        grid_result = grid_search.fit(X_train, y_train)

        # summarize result
        print('Best Accuracy Score (train)= {0:.3f} %'.format(grid_result.best_score_*100))
        print('Best Hyperparameters (train): {}'.format(grid_result.best_params_))

        # implement best model
        model = build_model(model_name=arg_dict['model_name'], grid_results=grid_result)
        model.fit(X_train, y_train)

        # create a "fake" history object for reporting results in a similar style to tensorflow models
        history = {}
        history['history'] = {}
        history['history']['accuracy'] = [grid_result.best_score_]
        history['history']['val_accuracy'] = [0]
        #history = AttrDict(history)

    else:

        # ONE HOT ENCODE LABELS ----------------------------------------------------------------------------------------
        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

        # TRAIN MODEL --------------------------------------------------------------------------------------------------
        input_shape = X.shape[1::]
        n_classes = len(np.unique(y_train, axis=0))
        epochs = arg_dict['epochs']
        batch_size = arg_dict['batch_size']

        callback = set_keras_callbacks(early_stop_patience=arg_dict['patience'])

        if args_dict['tune']:

            # initialize a tuner (here, RandomSearch). We use objective to specify the objective to select the best
            # models, and we use max_trials to specify the number of different models to try.

            tuner = kt.RandomSearch(hypermodel=build_model_tune,     # tuning options set in build_model_tune function
                                    objective='val_loss',
                                    max_trials=3,
                                    overwrite=True,
                                    directory='keras_tune_results')

            # start the search (always with early stop)
            tuner.search(X_train, y_train, validation_split=0.2, epochs=5, callbacks=callback)

            # extract best model
            model = tuner.get_best_models(num_models=1)[0]

            # print summary
            tuner.results_summary()

        else:
            model = build_model(model_name=args_dict['model_name'], n_classes=n_classes, input_shape=input_shape,
                                units=arg_dict['n_filts'], n_layers=args_dict['n_layers'], dropout=arg_dict['dropout'],
                                regularizer=args_dict['regularizer'], dropout_amt=arg_dict['dropout_amt'])

        # train or retrain (for tuned model)
        if arg_dict['early_stop']:
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,  validation_split=0.2, verbose=1,
                                callbacks=callback)
        else:
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)

    y_pred_train = model.predict(X_train)
    if np.ndim(y_pred_train) == 2:
        y_pred_train = np.argmax(y_pred_train, axis=1)
        y_train = np.argmax(y_train, axis=1)

    specificity_train = recall_score(y_train, y_pred_train, pos_label=0, average='weighted')  # https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
    clf_report_train = classification_report(y_train, y_pred_train, output_dict=True)
    cm_train = confusion_matrix(y_train, y_pred_train)

    # EVALUATE MODEL ON TEST SET --------------------------------------------------------------------------------------
    # - You should not tune your model to improve the score on the test set.
    # - see Chollet, Deep Learning with Python, p97
    cm_test = None
    clf_report_test = None
    specificity_test = None
    if args_dict['evaluate_on_test_set']:
        #todo : save the previously trained model to avoid running again?
        y_pred = model.predict(X_test)
    # else:
    #     y_pred = model.predict(X_train)
    #     y_test = y_train    # just used to simplify the code later, in the final run, evaluation is done on real test

        if np.ndim(y_pred) == 2:
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)

        specificity_test = recall_score(y_test, y_pred, pos_label=0, average='weighted')# https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
        clf_report_test = classification_report(y_test, y_pred, output_dict=True)
        cm_test = confusion_matrix(y_test, y_pred)

    # DISPLAY RESULTS FOR USER -----------------------------------------------------------------------------------------
    run_time = time.time() - start_time
    print_plot_save_results(arg_dict, history, X, y, channel_names, label_map, run_time, clf_report_train, cm_train,
                            specificity_train, clf_report_test, cm_test, specificity_test)


if __name__ == '__main__':
    # todo create a settings file
    import argparse
    parser = argparse.ArgumentParser(description="sample ml framework ")
    parser.add_argument('--data', default='keras_sample', choices={'keras_sample', 'HAR_sample'})

    # preprocessing arguments
    parser.add_argument('--channels', type=str, default='all', help='select all or a subset of channels by name')

    # arguments for choosing train/test split **************************************************************************
    parser.add_argument('--train_test_split_by_user', type=bool, default=False)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--validation_size', type=int, default=0.2)
    parser.add_argument('--evaluate_on_test_set', type=bool, default=False, help='only set to true after model tuning')

    # arguments for selection model type *******************************************************************************
    parser.add_argument('--model_name', type=str, default='cnn', help='see models.py for all choices',
                        choices={'cnn', 'svc'})

    # arguments for feature engineering ********************************************************************************
    parser.add_argument('--feature_selection', action='store_true')
    parser.add_argument('--feature_set', default='minimal', choices=['minimal', 'efficient', 'comprehensive'])

    # arguments for deep learning models only **************************************************************************
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_filts', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--early_stop', default=False, action='store_true')
    parser.add_argument('--dropout', default=False, action='store_true')
    parser.add_argument('--dropout_amt', type=float, default=0.3, help='dropout ratio, must be [0,1]')
    parser.add_argument('--regularizer', type=str, default=None, choices=['L1', 'L2', None], help='keras regularizer')
    parser.add_argument('--regularizer_amt', type=float, default=0.01)

    # tuning stuff
    parser.add_argument('--tune', default=False, action='store_true')

    args = parser.parse_args()
    args_dict = vars(args)

    # Check model type : It will return True if any of the substrings in substring_list is contained in string.
    substring_list = ['cnn', 'lstm']
    if any(substring in args_dict['model_name'] for substring in substring_list):  args_dict['model_type'] = 'signal'
    else:
        args_dict['model_type'] = 'features'

    main(args_dict)

