import os
import json
from matplotlib import pyplot as plt
import numpy as np
import datetime
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


def evaluate_model(y_true, y_pred):
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
    if n_classes > 2:
        y_true_cat = to_categorical(y_true)
        y_pred_cat = to_categorical(y_pred)
        metric_dict['AUC'] = metrics.roc_auc_score(y_true_cat, y_pred_cat, multi_class='ovr')  # one vs other
    else:
        metric_dict['AUC'] = metrics.roc_auc_score(y_true, y_pred)

    return metric_dict, cm


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


def create_results_dir(arg_dict):
    """ create unique directory for each run of a given dataset and model based on run time"""

    if arg_dict['trained_model_path'] is not None:
        result_dir_current_model= os.path.join(DIR_RESULTS, arg_dict['trained_model_path'])
    else:
        # create subdir for current model
        t = datetime.datetime.now().strftime('%Y-%m-%d_time_%H.%M.%S')
        model_name = arg_dict['model_name']
        if type(model_name) == list:
            model_name = model_name[0]
        current_model = arg_dict['database'] + '_' + model_name + '_date_' + t
        result_dir_current_model = os.path.join(DIR_RESULTS, current_model)
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


def save_args(arg_dict):
    """ saves argument dictionary to txt"""
    f = os.path.join(DIR_RESULTS, 'arguments.txt')
    rel_pth = os.path.relpath(f, DIR_ROOT)
    if arg_dict['verbose']:
        print('saving temporary arguments file to: {}'.format(rel_pth))
    with open(f, 'w') as file:
        file.write(json.dumps(arg_dict))


def load_args(verbose=False):
    """load argument dictionary from text file"""
    args_path = os.path.join(DIR_RESULTS, 'arguments.txt')
    f = open(args_path)
    rel_pth = os.path.relpath(args_path, DIR_ROOT)
    if verbose:
        print('loading arguments file from {}'.format(rel_pth))
    arg_dict = json.load(f)
    return arg_dict


def print_plot_save_results(arg_dict, model_eval_dict_train, model_eval_dict_test, cm_train, cm_test, history, y,
                            label_map, run_time, model, n_features):

    """print  plot, save results using keras history object and other arguments"""

    # set up results directory for current model
    results_dir_current_model = create_results_dir(arg_dict)
    n_classes = len(np.unique(y))
    print('\n******** RESULTS ************************************************************************************')
    print('*')
    print('* Data set description: --------------------------------------------------------')
    print('* Data set name:          {0}'.format(arg_dict['database']))
    print('* channel names:          {0}'.format(arg_dict['channel_names']))
    print('* Segment shape:          {0}'.format(arg_dict['segment_shape']))
    print('* n samples:              {0}'.format(len(y)))
    # print('* n channels:             {0}'.format(len(arg_dict['channel_names'])))
    print('* n classes:              {0}'.format(n_classes))
    for key, value in label_map.items():
        print('* lbl {} {:<17} {} ({:.1f}%)'.format(key, '(' + str(value) + '):',
                                                    np.count_nonzero(y == value),
                                                    np.count_nonzero(y == value) / len(y) * 100))

    print('* ')
    print('* Model details: -----------------------------------------------------------------')
    print('* Model:                  {0}'.format(arg_dict['model_name']))
    print('* type:                   {0}'.format(arg_dict['model_type']))
    print('* tune:                   {0}'.format(arg_dict['tune']))
    if arg_dict['tune']:
        print('* evaluation metric:              {0}'.format(arg_dict['evaluation_metric']))

    if arg_dict['model_type'] == 'signal':
        # todo : find better way to show all parser arguments for deep learning
        model.summary()
    else:
        print('* feature_selection:      {0}'.format(arg_dict['feature_selection']))
        print('* number of features:     {0}'.format(n_features))
    print('*')

    if history is not None:
        print('*')
        print('* Model training results ------------------------------------------------------')
        print('* training (total run) time: {0}'.format(run_time))
        print('* training set accuracy:     {0:.1f}%'.format(history.history['accuracy'][-1]*100))
        print('* validation set accuracy:   {0:.1f}%'.format(history.history['val_accuracy'][-1]*100))

    print('*')
    print('* Training set evaluation metrics ------------------------------------------------- ')
    if n_classes == 2:
        for k, v in model_eval_dict_train.items():
            if isinstance(v, list):
                v = v[0]
            print('* {:<45} {:>5.3f}%'.format(k + ':', v * 100))  # cute alignment

    else:
        for k, v in model_eval_dict_train.items():
            if not isinstance(v, float):
                for i, label in enumerate(label_map.keys()):
                    print('* {:<45} {:>6.3f}%'.format(k + ' (' + label + '):', v[i]*100))   # cute alignment
        print('*')
        for k, v in model_eval_dict_train.items():
            if isinstance(v, float):
                print('* {:<45} {:>6.3f}%'.format(k + ':', v * 100))  # cute alignment

    if model_eval_dict_test:
        print('*')
        print('* Test set evaluation metrics:------------------------------------------------- ')
        if n_classes ==2:
            for k, v in model_eval_dict_test.items():
                if isinstance(v, list):
                    v = v[0]
                print('* {:<45} {:>5.3f}%'.format(k + ':', v * 100))  # cute alignment

        else:
            for k, v in model_eval_dict_test.items():
                if not isinstance(v, float):
                    for i, label in enumerate(label_map.keys()):
                        print('* {:<45} {:>6.3f}%'.format(k + ' (' + label + '):', v[i] * 100))  # cute alignment
            print('*')
            for k, v in model_eval_dict_test.items():
                if isinstance(v, float):
                    print('* {:<45} {:>5.3f}%'.format(k + ':', v * 100))  # cute alignment

    print('*')

    # save train results -----------------------------------------------------------------------------------------------
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
            print('* results saved to: {}'.format(results_dir_current_model))

        with open(os.path.join(results_dir_current_model, 'arguments.txt'), 'w') as file:
            file.write(json.dumps(arg_dict))

        if model_eval_dict_train:
            with open(os.path.join(results_dir_current_model, 'train_results.txt'), 'w') as file:
                file.write(json.dumps(model_eval_dict_train))
            plt_cm = plot_confusion_matrix(cm_train, label_map.keys(),
                                           title='Train CM: {0} channels, AUC {1:.1f}%'
                                           .format(len(arg_dict['channel_names']),
                                                   model_eval_dict_train['AUC'] * 100))
            plt_cm.savefig(os.path.join(results_dir_current_model, 'train_cm.png'))

        if arg_dict['model_type'] == 'signal':
            plt_history = plot_train_val_acc_loss(history)
            plt_history.savefig(os.path.join(results_dir_current_model, 'train_val_acc_loss.png'))

        # save model
        model_pth = os.path.join(results_dir_current_model, 'model')
        if arg_dict['model_type'] == 'signal':
            if model is not None:
                model.save(model_pth + '.h5')
        else:
            pickle.dump(model, open(model_pth + '.pickle', "wb"))

        # save history file
        if history is not None:
            with open(os.path.join(results_dir_current_model, 'history.pickle'), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

    if model_eval_dict_test:
        with open(os.path.join(results_dir_current_model, 'test_results.txt'), 'w') as file:
            file.write(json.dumps(model_eval_dict_test))
        plt_cm = plot_confusion_matrix(cm_test, label_map.keys(),
                                       title='Test CM: {0} channels, AUC {1:.1f}%'
                                       .format(len(arg_dict['channel_names']),
                                               model_eval_dict_test['AUC'] * 100))
        plt_cm.savefig(os.path.join(results_dir_current_model, 'test_cm.png'))

    print('*')
    print('* Date: {}\n'.format(datetime.datetime.now()))
    print('*****************************************************************************************************\n')


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


def print_plot_feature_class_relation(X, y_cat):
    """ visualize differences between data classes """

    # categorical to encoded y
    y = np.argmax(y_cat, axis=1)

    # determine number of classes
    n_classes = len(np.unique(y))

    t = np.linspace(0, 1, len(X[0, :, 0]))
    for n_class in range(0, n_classes):
        stk = []
        for i in range(0, len(y)):
            yi = y[i]
            if yi == n_class:
                stk.append(X[i, :, :])
        stk_arr = np.array(stk)
        plt.plot(t, stk_arr[:, :, 0])
        plt.show()


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
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    return plt



