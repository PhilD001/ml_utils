import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import os

from processing.processing import load_database
from machine_learning.models import build_model, tune_keras_model, set_keras_callbacks, tune_sklearn_model, get_mdl_type
from utils.utils import print_plot_save_results, eval_model, load_trained_model, save_args, subject_wise_split, \
    batch_results, create_results_dir
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
    start_time = datetime.now()
    results_fld = create_results_dir(start_time, arg_dict)

    # 1 - LOAD DATA ----------------------------------------------------------------------------------------------------
    # - shape of data should be (samples, frames, channels), see Deep Learning with Python p 36
    X, y, users, ch_names, conditions, meta_data = load_database(arg_dict)

    # encode labels: for binary problem, make sure the positive class (1) is what you are studying, see discussion here:
    # https://medium.com/@asimango/the-positive-class-what-should-it-be-in-a-machine-learning-binary-classification-problem-36c316da1127
    le = LabelEncoder()
    y = le.fit_transform(y)
    label_map = {l: i for i, l in enumerate(le.classes_)}

    # CROSS VALIDATION LOOP
    for k in range(0, arg_dict['cross_val_num']):
        random_state = arg_dict['random_state'] + k
        loop_dir = os.path.join(DIR_RESULTS, results_fld, 'cross_val_it_' + str(k+1) + '_seed_' + str(random_state))
        os.mkdir(loop_dir)
        loop_time = datetime.now()
        print('running iteration {} with random seed {}'.format(k + 1, random_state))

        # 2 - TRAIN/TEST SPLIT DATA ------------------------------------------------------------------------------------
        if users is None and arg_dict['train_test_split_by_user']:
            arg_dict['train_test_split_by_user'] = False
        by_subject = arg_dict['train_test_split_by_user']
        test_size = arg_dict['test_ratio']
        X_train, X_test, y_train, y_test, users_train, users_test = subject_wise_split(X, y, users,
                                                                                       subject_wise=by_subject,
                                                                                       test_size=test_size,
                                                                                       random_state=random_state)

        # 3 - DATA AUGMENTATION (OPTIONAL) -----------------------------------------------------------------------------
        if arg_dict['augment_data'] != 'none':
            raise NotImplementedError('currently augmentation is performed on features only')

        # 4 - SCALE DATA -----------------------------------------------------------------------------------------------
        # -In the interest of preventing information about the distribution of the test set leaking into your model, you
        #  should fit the scaler on your training data only, then standardise training and test sets with that scaler.
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

        # 5 - SELECT MODEL TYPE (FEATURE VS SIGNAL) --------------------------------------------------------------------
        if arg_dict['model_type'] == 'features':
            from feature_engineering.features import extract_features, feature_selection, feature_clean
            # 5 A: FEATURE-BASED MODEL ---------------------------------------------------------------------------------
            freq = meta_data['freq']
            X_test, feat_names = extract_features(X_test, ch_names, freq, feature_set='test',
                                                  force_recompute=arg_dict['force_recompute_features'],
                                                  verbose=arg_dict['verbose'])
            X_train, _ = extract_features(X_train, ch_names, freq, feature_set='train',
                                          force_recompute=arg_dict['force_recompute_features'],
                                          verbose=arg_dict['verbose'])
            n_original_features = len(feat_names)

            # basic cleaning for nans and inf (always done)
            X_train, X_test, feat_names = feature_clean(X_train, X_test, feat_names, verbose=arg_dict['verbose'])

            # modifiable arguments: corr_threshold, variance_threshold (default used here)
            feature_selection_method = arg_dict['feature_selection_method']
            if feature_selection_method == 'rules':
                cor = arg_dict['correlation_threshold']
                X_train, X_test, feat_names = feature_selection(X_train, X_test, feat_names,
                                                                correlation_threshold=cor,
                                                                verbose=arg_dict['verbose'])

            # Feature augmentation via SMOTE
            if arg_dict['feature_augmentation']:
                raise NotImplementedError

            # Train model (or load pre-trained model)
            if arg_dict['trained_model_path']:
                model, arg_dict, history = load_trained_model(arg_dict)
            else:
                if arg_dict['tune']:
                    eval_metric = arg_dict['evaluation_metric']
                    if feature_selection_method == 'during_training':
                        feature_select = True
                    else:
                        feature_select = False
                    model, feat_names_tune, feat_indices = tune_sklearn_model(X_train, y_train, feat_names, eval_metric,
                                                                              participants=users_train,
                                                                              feature_select=feature_select,
                                                                              verbose=arg_dict['verbose'])

                    # update X_train, X_test with correct indices
                    if feat_indices is not None:
                        X_train = X_train[:, feat_indices]
                        X_test = X_test[:, feat_indices]
                        feat_names = feat_names_tune

                else:
                    model = build_model(arg_dict['model_name'])

                model.fit(X_train, y_train)
                history = None

        else:
            from tensorflow import keras

            # 5 B: SIGNAL BASED MODEL ----------------------------------------------------------------------------------
            n_features, n_original_features, feat_names = None, None, None

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
                    # initialize a tuner (here, RandomSearch). We use objective to specify the objective to select the
                    # best models, and we use max_trials to specify the number of different models to try.
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

                    model = build_model(model_name=arg_dict['model_name'][0], n_classes=n_classes,
                                        input_shape=input_shape,
                                        units_input=units_input, units_inner=units_inner, units_output=units_output,
                                        n_layers=n_layers, dropout=dropout, dropout_amt=dropout_amt,
                                        regularizer=regularizer, kernel_size=kernel_size,
                                        post_conv_layer=post_conv_layer)

                # train or retrain (for tuned model)
                history = model.fit(X_train, y_train, batch_size=arg_dict['batch_size'], epochs=arg_dict['epochs'],
                                    validation_split=arg_dict['validation_ratio'], verbose=1, callbacks=callback)

        # 6 - EVALUATE TRAINED MODEL -----------------------------------------------------------------------------------
        y_pred_train = model.predict(X_train)
        y_pred_prob_train = model.predict_proba(X_train)[:, 1]

        model_eval_dict_train, cm_train, fpr, tpr, threshold = eval_model(y_train, y_pred_train, y_pred_prob_train)

        # 7 - EVALUATE MODEL ON TEST SET (OPTIONAL) --------------------------------------------------------------------
        # - You should not tune your model to improve the score on the test set: Chollet, Deep Learning with Python, p97
        model_eval_dict_test = {}
        cm_test = None
        if arg_dict['evaluate_on_test_set']:
            y_pred_test = model.predict(X_test)
            y_pred_prob_test = model.predict_proba(X_test)[:, 1]  # todo: check if this is ok for different class numbs
            model_eval_dict_test, cm_test, fpr, tpr, threshold = eval_model(y_test, y_pred_test, y_pred_prob_test)

        # 8 - PRINT PLOT, SAVE RESULTS AND MODELS ----------------------------------------------------------------------
        print_plot_save_results(arg_dict, model_eval_dict_train, model_eval_dict_test, cm_train, cm_test, history, y,
                                label_map, loop_time, model, n_original_features, feat_names, fpr, tpr, loop_dir,
                                X_test, y_test)

    # 9 - SUMMARIZE RESULTS ACROSS ALL FOLDS ---------------------------------------------------------------------------
    if arg_dict['cross_val_num'] > 1:
        batch_results(results_fld)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Cough Detection Project")

    # Data arguments
    parser.add_argument('--database', default='hx_cough_db', choices={'keras_sample', 'HAR_sample', 'hx_cough_db'})
    parser.add_argument('--channel_names', nargs='+', default=['all'])
    parser.add_argument('--augment_data', default='none', choices={'warp', 'warp_jitter', 'jitter', 'none'},
                        help='see processing.data_augment, not implemented')

    # General arguments
    parser.add_argument('--verbose', default=False, action='store_true', help='use flag to print information to screen')
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE, help='use to change random seed')
    parser.add_argument('--cross_val_num', default=1, type=int, help='iterate cross_val_num times')

    # Data processing
    parser.add_argument('--scale_signal', default=False, action='store_true')

    # arguments for choosing train/test split **************************************************************************
    parser.add_argument('--train_test_split_by_user', default=True, action='store_true')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--validation_ratio', type=int, default=0.2)
    parser.add_argument('--evaluate_on_test_set', default=False, action='store_true', help='after train')
    parser.add_argument('--trained_model_path', type=str, default=None, help='path to pre-trained model in results dir')

    # arguments for selection model type *******************************************************************************
    parser.add_argument('--model_name', nargs='+', default=['random_forest'], help='see models.py')

    # arguments for feature engineering ********************************************************************************
    parser.add_argument('--feature_selection_method', type=str, default='rules', choices=['rules', 'during_tuning'])
    parser.add_argument('--save_features', default=True, action='store_true', help='save to disk for quick reload')
    parser.add_argument('--force_recompute_features', default=False, action='store_true', help='ignore save, recompute')
    parser.add_argument('--correlation_threshold', type=float, default=0.95, help='corr thresh used in feature select')
    parser.add_argument('--feature_augmentation', default=False, action='store_true')
    parser.add_argument('--feature_augmentation_amount', type=float, default=1.6)

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
    parser.add_argument('--post_conv_layer', type=str, default='batch_normalize',
                        choices={'batch_normalize', 'max_pool'})

    # Model arguments
    parser.add_argument('--tune', default=False, action='store_true', help='automatic parameter tuning during training')
    parser.add_argument('--tune_and_select', default=False, action='store_true', help='automatic feature selection during tuning')
    parser.add_argument('--evaluation_metric', default='accuracy', choices={'auc', 'accuracy', 'precision', 'F1'},
                        help='metric for tuning. F1 doesnt work')

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['model_type'] = get_mdl_type(args_dict['model_name'])
    main(args_dict)
