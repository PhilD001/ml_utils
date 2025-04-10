import os
import json
from itertools import combinations
from datetime import datetime
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from attributedict.collections import AttributeDict
from definitions import DIR_RESULTS, DIR_ROOT


def subject_wise_split(x, y, participant, subject_wise=True, test_size=0.10, random_state=42):
    """ Split data into train and test sets via an inter-subject scheme, see:
    Shah, V., Flood, M. W., Grimm, B., & Dixon, P. C. (2022). Generalizability of deep learning models for predicting outdoor irregular walking surfaces.
    Journal of Biomechanics, 139, 111159. https://doi.org/10.1016/j.jbiomech.2022.111159

    Arguments:
        x: nd.array, feature space
        y: nd.array, label class
        participant: nd.array, participant associated with each row in x and y
        subject_wise: bool, choices {True, False}, default = True. True = subject-wise split approach, False random-split
        test_size: float, number between 0 and 1. Default = 0.10. percentage spilt for test set.
        random_state: int. default = 42. Seed selector for numpy random number generator.
    Returns:
        x_train: nd.array, train set for feature space
        x_test: nd.array, test set for feature space
        y_train: nd.array, train set label class
        y_test: nd.array, test set label class
        subject_train: nd.array[string], train set for participants by row of input data
        subjects_test: nd.array[string[, test set for participants by row of input data
    """
    if type(participant) == list:
        participant = np.asarray(participant, dtype=np.float32)

    np.random.seed(random_state)
    if subject_wise:
        uniq_parti = np.unique(participant)
        num = np.round(uniq_parti.shape[0] * test_size).astype('int64')
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
        num = np.round(participant.shape[0] * test_size).astype('int64')
        test_index = index[0:num]
        train_index = index[num:]

    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    subject_train = participant[train_index]
    subject_test = participant[test_index]

    return x_train, x_test, y_train, y_test, subject_train, subject_test


def eval_model(y_true, y_pred, y_pred_prob=None):
    """ util to group different evaluation metrics

    NOTE: From sklearn documentation : in binary classification, recall of the positive class is also known as
          “sensitivity”; recall of the negative class is “specificity”.
          https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """

    # check if values are probabilities, then convert
    if np.ndim(y_pred) == 2:
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)

    # get number of classes to determine if binary or multiclass problem
    n_classes = len(np.unique(y_true))

    # confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    # get metrics from confusion matrix
    metric_dict = get_metrics_from_cm(cm)

    # get other metrics from sklearn
    metric_dict['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    metric_dict['accuracy_balanced'] = metrics.balanced_accuracy_score(y_true, y_pred)
    metric_dict['F1'] = metrics.f1_score(y_true, y_pred, average='weighted')
    if n_classes == 2:
        metric_dict['Precision'] = metrics.precision_score(y_true, y_pred)
    else:
        metric_dict['Precision'] = metrics.precision_score(y_true, y_pred, average='weighted')

    # compute false positive rates and true positive rates based on thresholds
    if n_classes == 2:
        if y_pred_prob is not None:
            fpr, tpr, thresh = metrics.roc_curve(y_true, y_pred_prob, drop_intermediate=False)
        else:
            fpr, tpr, thresh = metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
    else:
        fpr, tpr, thresh = None, None, None

    if n_classes > 2:
        y_true_cat = to_categorical(y_true)
        y_pred_cat = to_categorical(y_pred)
        metric_dict['auc'] = metrics.roc_auc_score(y_true_cat, y_pred_cat, multi_class='ovr')  # one vs other
    else:
        metric_dict['auc'] = metrics.roc_auc_score(y_true, y_pred)

    return metric_dict, cm, fpr, tpr, thresh


def sensitivity_given_specificity(fpr, tpr, thresholds, desired_specificity=0.9, verbose=False):
    """ compute sensitivity given a required specificity """

    # Calculate specificity (which is 1 - FPR)
    specificity = 1 - fpr

    # Find the index where specificity is closest to the desired value
    idx = (np.abs(specificity - desired_specificity)).argmin()

    # Get the corresponding threshold
    optimal_threshold = thresholds[idx]

    # Get the corresponding sensitivity (true positive rate)
    associated_sensitivity = tpr[idx]

    if verbose:
        print('Optimal Threshold for Specificity {} = {:.3f}'.format(desired_specificity, optimal_threshold))
        print('Associated Sensitivity (TPR) =  {:.3f}'.format(associated_sensitivity))

    return associated_sensitivity


def get_metrics_from_cm(cm):
    """ calculates metrics directly from confusion matrix, see:
    https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    """

    # extract all variables
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # compute metrics
    metric_dict = {}

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    metric_dict['sensitivity'] = list(TPR)
    # metric_dict['sensitivity_average'] = np.mean(TPR)  # calculated by Phil D, seems reasonable

    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    metric_dict['specificity'] = list(TNR)
    # metric_dict['specificity_average'] = np.mean(TNR)  # calculated by Phil D, seems reasonable

    # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # metric_dict['positive predictive value'] = list(PPV)
    #
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # metric_dict['negative predictive value'] = list(NPV)
    #
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # metric_dict['false positive rate'] = list(FPR)
    #
    # # False negative rate
    # FNR = FN / (TP + FN)
    # metric_dict['false negative rate'] =  list(FNR)
    #
    # # False discovery rate
    # FDR = FP / (TP + FP)
    # metric_dict['false discovery rate'] = list(FDR)

    # Overall accuracy for each class
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    # metric_dict['accuracy'] = list(ACC)

    # balanced accuracy of sklearn
    # defined as the average of recall obtained on each class, see:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
    # metric_dict['accuracy_balanced'] = BACC

    return metric_dict


def create_results_dir(arg_dict, res_sfld):
    """ create unique directory for each run of a given dataset and model based on run time inside res_sfld"""

    if arg_dict['trained_model_path'] is not None:
        result_dir_current_model = os.path.join(DIR_RESULTS, res_sfld, arg_dict['trained_model_path'])
    else:
        # create subdir for current model
        t = datetime.datetime.now().strftime('%Y-%m-%d_time_%H.%M.%S')
        model_name = arg_dict['model_name']
        if type(model_name) == list:
            model_name = model_name[0]
        current_model = arg_dict['database'] + '_' + model_name + '_date_' + t
        result_dir_current_model = os.path.join(DIR_RESULTS, res_sfld, current_model)
        os.mkdir(result_dir_current_model)
    return result_dir_current_model


def load_trained_model(arg_dict):
    """load pretrained model from file """
    # todo : check saving location

    # extract existing arg dict settings
    trained_model_path = arg_dict['trained_model_path']
    evaluate_on_test_set = arg_dict['evaluate_on_test_set']

    model_pth = os.path.join(DIR_RESULTS, trained_model_path, 'model')
    if arg_dict['verbose']:
        print('loading pre-trained model from {}'.format(trained_model_path))
    if arg_dict['model_type'] == 'signal':
        model = tf.keras.models.load_model(model_pth + '.h5')
        with open(os.path.join(DIR_RESULTS, trained_model_path, 'history.pickle'), "rb") as file_pi:
            hist = pickle.load(file_pi)
            history = AttributeDict({'history': hist})
    elif arg_dict['model_type'] == 'features':
        model = pickle.load(open(model_pth + '.pickle', 'rb'))
        history = None
    else:
        raise ValueError('Unknown model type {}'.format(arg_dict['model_type']))

    # get saved/original arg_dict and overwrite with possible new values
    arg_dict = json.load(open(os.path.join(DIR_RESULTS, trained_model_path, 'arguments.txt')))
    arg_dict['evaluate_on_test_set'] = evaluate_on_test_set
    arg_dict['trained_model_path'] = trained_model_path

    return model, arg_dict, history


def plot_strips(X, y, label_map, channel_names, pth_save, n_examples=5):
    """plot n strips in database for each label """

    n_channels = X.shape[1]
    labels = np.unique(y)
    for label in labels:
        res = dict((v, k) for k, v in label_map.items())
        indx = [i for i in range(len(y)) if y[i] == label]

        if len(indx) < n_examples:
            n_examples = len(indx)

        for i in range(0, n_examples):
            current_strip = X[indx[i], :, :]
            current_label = y[indx[i]]
            current_label_name = res[current_label]

            fig, axs = plt.subplots(n_channels)
            for j in range(0, n_channels):
                current_strip_by_channel = current_strip[j, :]
                axs[j].plot(current_strip_by_channel)
                axs[j].set_ylabel(channel_names[j])
            fig.suptitle('Sample signals for label {} ({})'.format(current_label, current_label_name))
            fig.savefig(os.path.join(pth_save, "sample_strip_{}_label_{}_{}".format(i, current_label,
                                                                                    current_label_name) + ".png"),
                        dpi=600)


def plot_by_label(X, y, y_labels=None, n_examples=10):
    """ plots signal by label """
    n_plots = len(np.unique(y))
    fig, axs = plt.subplots(n_plots)
    for n in range(0, n_plots):
        indx = np.where(y == n)[0]
        if len(indx) < n_examples:
            n_examples = len(indx)
        X_by_lbl = X[indx[0:n_examples]]
        X_by_lbl_reshaped = np.reshape(X_by_lbl, (np.shape(X_by_lbl)[0] * np.shape(X_by_lbl)[1], np.shape(X_by_lbl)[2]))

        axs[n].plot(X_by_lbl_reshaped)
        xposition = np.linspace(np.shape(X_by_lbl)[1], np.shape(X_by_lbl)[0] * np.shape(X_by_lbl)[1],
                                np.shape(X_by_lbl)[0])
        for xc in xposition:
            axs[n].axvline(x=xc, color='k', linestyle='--')
        if y_labels:
            axs[n].set_title('label {}'.format(y_labels[n]))
        else:
            axs[n].set_title('label {}'.format(n))
    fig.suptitle('Sample signals by label')
    plt.show()


def save_args(arg_dict):
    """ saves argument dictionary to txt"""
    f = os.path.join(DIR_RESULTS, 'arguments_temp.txt')
    rel_pth = os.path.relpath(f, DIR_ROOT)
    if arg_dict['verbose']:
        print('saving temporary arguments file to: {}'.format(rel_pth))
    with open(f, 'w') as file:
        file.write(json.dumps(arg_dict))


def load_args(verbose=False):
    """load argument dictionary from text file"""
    args_path = os.path.join(DIR_RESULTS, 'arguments_temp.txt')
    f = open(args_path)
    rel_pth = os.path.relpath(args_path, DIR_ROOT)
    if verbose:
        print('loading arguments file from {}'.format(rel_pth))
    arg_dict = json.load(f)
    return arg_dict


def print_plot_save_results(arg_dict, model_eval_dict_train, model_eval_dict_test, cm_train, cm_test, history, y,
                            label_map, loop_time, model, n_original_features, feat_names, fpr, tpr, loop_dir,
                            x_test=None, y_test=None):
    """print  plot, save results using keras history object and other arguments"""

    # End time
    end_time = datetime.now()

    # Calculate elapsed time
    elapsed = end_time - loop_time
    total_seconds = int(elapsed.total_seconds())

    # Convert to hours, minutes, seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print('Elapsed time: {:02d}h:{:02d}m:{:02d}s'.format(hours, minutes, seconds))

    # extract number of clases
    n_classes = len(np.unique(y))

    # save number of features
    if feat_names is not None:
        n_features = len(feat_names)

    # set up results directory for current model
    model_name = arg_dict['model_name']
    if type(model_name) == list:
        model_name = model_name[0]

    print('\n******** RESULTS ***************************************')
    print('*')
    print('* Data set description: --------------------------------')
    print('* Data set name:                       {0}'.format(arg_dict['database']))
    print('* n frames per strip:                  {0}'.format(arg_dict['segment_shape'][0]))
    print('* n channels:                          {0}'.format(arg_dict['segment_shape'][1]))
    print('* n samples:                           {0}'.format(len(y)))
    print('* n classes:                           {0}'.format(n_classes))
    for key, value in label_map.items():
        print('* label {:<29} {:>5.1f}% ({})'.format(key + ' (' + str(value) + '):',
                                                     np.count_nonzero(y == value) / len(y) * 100,
                                                     np.count_nonzero(y == value)))

    print('* ')
    print('* Model details: ----------------------------------------')
    print('* Model:                               {0}'.format(model_name))
    print('* type:                                {0}'.format(arg_dict['model_type']))
    print('* evaluation metric:                   {0}'.format(arg_dict['evaluation_metric']))
    print('* tune:                                {0}'.format(arg_dict['tune']))

    if arg_dict['model_type'] == 'features':
        print('* feature_selection_method:                   {0}'.format(arg_dict['feature_selection_method']))
        print('* number of features (total):          {0}'.format(n_original_features))
        if arg_dict['feature_selection_method']:
            print('* number of features (reduced):        {0}'.format(n_features))
    print('*')

    if history is not None:
        print('*')
        print('* Model training results -----------------------------------')
        # print('* training (total run) time:      {0}'.format(run_time))
        print('* training set accuracy:          {0:.1f}%'.format(history.history['accuracy'][-1] * 100))
        print('* validation set accuracy:        {0:.1f}%'.format(history.history['val_accuracy'][-1] * 100))

    print('*')
    print('* Training set evaluation metrics -------------------------')

    for k, v in model_eval_dict_train.items():
        if not isinstance(v, float):
            for i, label in enumerate(label_map.keys()):
                print('* {:<35} {:>5.3f}%'.format(k + ' (' + label + '):', v[i] * 100))  # cute alignment
    print('*')
    for k, v in model_eval_dict_train.items():
        if isinstance(v, float):
            print('* {:<35} {:>5.3f}%'.format(k + ':', v * 100))  # cute alignment

    if model_eval_dict_test:
        print('*')
        print('* Test set evaluation metrics:------------------------------')

        for k, v in model_eval_dict_test.items():
            if not isinstance(v, float):
                for i, label in enumerate(label_map.keys()):
                    print('* {:<35} {:>6.3f}%'.format(k + ' (' + label + '):', v[i] * 100))  # cute alignment
        print('*')
        for k, v in model_eval_dict_test.items():
            if isinstance(v, float):
                print('* {:<35} {:>5.3f}%'.format(k + ':', v * 100))  # cute alignment

    print('*')

    # save train results ----------------------------------------------------
    if arg_dict['trained_model_path'] is not None:
        if arg_dict['verbose']:
            print('* Training results previously saved to: {}'.format(arg_dict['trained_model_path']))
            print('* arguments: {}'.format(os.path.join(arg_dict['trained_model_path'], 'arguments.txt')))
            if history:
                print('* history: {}'.format(os.path.join(arg_dict['trained_model_path'], 'history.txt')))
            print('* train results: {}'.format(os.path.join(arg_dict['trained_model_path'], 'train_results.txt')))
            print('* train cm: {}'.format(os.path.join(arg_dict['trained_model_path'], 'train_cm.png')))

    else:
        if arg_dict['verbose']:
            print('* results saved to: {}'.format(loop_dir))

        with open(os.path.join(loop_dir, 'arguments.txt'), 'w') as file:
            file.write(json.dumps(arg_dict))

        if model_eval_dict_train:
            with open(os.path.join(loop_dir, 'train_results.txt'), 'w') as file:
                file.write(json.dumps(model_eval_dict_train))
            plt_cm = plot_confusion_matrix(cm_train, label_map.keys(),
                                           title='Train CM: {0} channels, auc {1:.1f}%'
                                           .format(len(arg_dict['channel_names']),
                                                   model_eval_dict_train['auc'] * 100))
            plt_cm.savefig(os.path.join(loop_dir, 'train_cm.png'))

        if arg_dict['model_type'] == 'signal':
            plt_history = plot_train_val_acc_loss(history)
            plt_history.savefig(os.path.join(loop_dir, 'train_val_acc_loss.png'))

        if arg_dict['model_name'][0] in ['RandomForestClassifier', 'random_forest', 'DecisionTree']:

            if arg_dict['tune']:
                importances = model.named_steps['classifier'].feature_importances_
                std = np.std([tree.feature_importances_ for tree in model.named_steps['classifier'].estimators_],
                             axis=0)
            else:
                importances = model.feature_importances_
                std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            if len(importances) > 20:
                importances = importances[0:20]
                std = std[0:20]
            feat_names = feat_names[0:len(importances)]
            plt_imp, df = plot_importances(importances, std, feat_names)
            plt_imp.savefig(os.path.join(loop_dir, 'feature_importances.png'))
            df.iloc[:, 0].to_csv(os.path.join(loop_dir, 'feature_importances.txt'), index=False,
                                 header=False)

        # save model
        model_pth = os.path.join(loop_dir, 'model')
        if arg_dict['model_type'] == 'signal':
            if model is not None:
                model.save(model_pth + '.h5')
        else:
            pickle.dump(model, open(model_pth + '.pickle', "wb"))

        # save history file
        if history is not None:
            with open(os.path.join(loop_dir, 'history.pickle'), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

    if model_eval_dict_test:

        # save test results text file
        with open(os.path.join(loop_dir, 'test_results.txt'), 'w') as file:
            file.write(json.dumps(model_eval_dict_test))

        # plot confusion matrix
        metric = arg_dict['evaluation_metric']
        plt_cm = plot_confusion_matrix(cm_test, label_map.keys(),
                                       title='Test CM: {0} channels, {1} {2:.1f}%'
                                       .format(len(arg_dict['channel_names']), metric,
                                               model_eval_dict_test[metric] * 100))
        plt_cm.savefig(os.path.join(loop_dir, 'test_cm.png'))

        # plot ROC AUC curve
        if n_classes == 2:
            plt = plot_roc_auc(fpr, tpr, model_eval_dict_test['auc'])
            plt.savefig(os.path.join(loop_dir, 'roc_auc_curve.png'))

            # plot precision-recall AUC curve
            disp = metrics.PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            # Add grid
            disp.ax_.grid(True)
            # Show legend
            disp.ax_.legend()
            disp.figure_.savefig(os.path.join(loop_dir, 'prec_recall_curve.png'), format="png",
                                 dpi=300)

    print('* Date: {}\n'.format(datetime.now()))
    print('************************************************************\n')


def plot_roc_auc(fpr, tpr, auc=None):
    """ plot ROC AUC Curve
    see https://www.datatechnotes.com/2019/11/how-to-create-roc-curve-in-python.html
    """

    if auc is None:
        auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = {0:.3f}%)'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    return plt


def plot_precision_recall_auc(fpr, tpr, auc=None):
    """ plot precision recall AUC Curve
    see https://www.datatechnotes.com/2019/11/how-to-create-roc-curve-in-python.html
    """

    if auc is None:
        auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = {0:.3f}%)'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    return plt


def plot_importances(importances, std, feature_names=None):
    """ plot feature importances"""
    if feature_names is None:
        feature_names = [f"feature {i}" for i in range(importances.shape[1])]

    # create a dataframe
    data = {'feature names': feature_names,
            'importance': importances}
    df = pd.DataFrame(data)

    # sort the DataFrame by descending order
    df = df.sort_values('importance', ascending=False)

    # create a bar plot of the sorted DataFrame
    _, _ = plt.subplots()
    plt.bar(df['feature names'], df['importance'])
    plt.xticks(rotation=90)
    plt.tight_layout()
    return plt, df


def plot_train_val_acc_loss(history):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(history.history['accuracy'], label='accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    return plt


def tensor2dataframe(x, users, channel_names):
    """ takes data in tensor format and transforms to data frame """

    tensor_list = []
    time_list = []
    user_list = []
    trial_list = []
    for i in range(0, x.shape[0]):
        tensor_list.append(x[i, :, :])
        time_list.append(np.arange(0, x.shape[1]))
        user_list.append([int(users[i])] * x.shape[1])
        trial_list.append([int(i)] * x.shape[1])

    df = pd.DataFrame(np.vstack(tensor_list), columns=channel_names)
    df['id'] = np.hstack(user_list)
    df['time'] = np.hstack(time_list)
    df['trial'] = np.hstack(trial_list)

    return df


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Increase bottom padding
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def chunk_generator(data, frames_per_chunk):
    """ split data into chunks on length frames_per_chunk. Last chunk may be of length < than frames_per_chunk """
    for i in range(0, len(data), frames_per_chunk):
        yield data[i:i + frames_per_chunk]


def rename_duplicate_columns(df, verbose=False):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            if verbose:
                print('renaming feature {}'.format(newitem))
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df


def batch_results(fld):
    """ iterates through all results folders in fld to summarize data"""
    # Dictionary to store collected metrics
    metrics = {}
    feature_sets = []  # List to store sets of important features
    results_dir = os.path.join(DIR_RESULTS, fld)
    # Walk through all subdirectories inside results_dir
    for root, _, files in os.walk(results_dir):
        if os.path.exists(os.path.join(results_dir, root, 'test_results.txt')):
            result_file = os.path.join(results_dir, root, "test_results.txt")
            print('extracting results from: {}'.format(result_file))
            with open(result_file, "r") as f:
                data = json.load(f)  # Load JSON data

            # Process each metric
            for key, value in data.items():
                if isinstance(value, list):
                    value = value[0]  # Take only the first value
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

        # Process feature_importances.txt if it exists
        if "feature_importances.txt" in files:
            feature_file = os.path.join(results_dir, root, "feature_importances.txt")

            with open(feature_file, "r") as f:
                features = {line.strip() for line in f.readlines() if line.strip()}
                feature_sets.append(features)  # Store as a set for Jaccard similarity

    # Compute and print mean and standard deviation for each metric
    output_path = os.path.join(DIR_RESULTS, fld, 'metrics_summary.txt')
    with open(output_path, 'w') as f:
        f.write("Metrics Summary ---------------------\n")
        print("Metrics Summary ---------------------")
        for key, values in metrics.items():
            avg = np.mean(values)
            std = np.std(values, ddof=1)  # Sample standard deviation (N-1)
            line = '{}: {:.2f} ({:.2f})'.format(key, avg * 100, std * 100)
            print(line)
            f.write(line + '\n')

    if feature_sets:
        output_path = os.path.join(DIR_RESULTS, fld, 'jaccard_analysis.txt')
        with open(output_path, 'w') as f:
            f.write("Jaccard Analysis summary ---------------------\n")

            # Compute Jaccard similarity between all feature sets
            total_similarity = 0  # Initialize a variable to store the total similarity
            num_pairs = 0  # Initialize a counter for the number of pairs

            # Compute Jaccard similarities for all pairs
            for i, set1 in enumerate(feature_sets):
                for j, set2 in enumerate(feature_sets):
                    if i < j:  # Only calculate for unique pairs (avoid duplicates)
                        similarity = jaccard_similarity(set1, set2)
                        total_similarity += similarity
                        num_pairs += 1

            # Compute the average Jaccard similarity
            average_similarity = total_similarity / num_pairs if num_pairs > 0 else 0
            line = 'Average Jaccard Similarity {:.3f}%'.format(average_similarity * 100)
            print(line)
            f.write(line + '\n')

            # extract most representative list based on Jaccard similarity
            most_rep_feat_set = find_representative_list(feature_sets)
            f.write('Most representative feature set:' + '\n')
            for item in most_rep_feat_set:
                f.write(item + "\n")
            print('most representative feature set {}'.format(most_rep_feat_set))


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def find_representative_list(lists):
    sets = [set(lst) for lst in lists]
    scores = defaultdict(float)

    for i, s1 in enumerate(sets):
        for j, s2 in enumerate(sets):
            if i != j:
                scores[i] += jaccard_similarity(s1, s2)

    # Return the list with the highest total similarity score
    best_index = max(scores, key=scores.get)
    return lists[best_index]


def create_results_dir(start_time, arg_dict):
    # set up root results directory (first run only)
    if not os.path.exists(DIR_RESULTS):
        os.mkdir(DIR_RESULTS)
    # set up sub folder for each specific run
    formatted_time = start_time.strftime('%Y-%m-%d_%H_%M_%S')
    res_sfld = arg_dict['database'] + '_' + arg_dict['model_name'][0] + '_' + arg_dict['evaluation_metric'] + '_' + \
               '_tuned_' + str(arg_dict['tune']) + '_date_' + formatted_time
    os.mkdir(os.path.join(DIR_RESULTS, res_sfld))
    return res_sfld