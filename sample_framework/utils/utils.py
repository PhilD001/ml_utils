import os
import json
from matplotlib import pyplot as plt
import numpy as np
import datetime


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
            fig.savefig(os.path.join(pth_save, "sample_strip_{}_label_{}_{}".format(i, current_label, current_label_name) + ".png"), dpi=600)


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
        xposition = np.linspace(np.shape(X_by_lbl)[1],np.shape(X_by_lbl)[0]*np.shape(X_by_lbl)[1], np.shape(X_by_lbl)[0])
        for xc in xposition:
            axs[n].axvline(x=xc, color='k', linestyle='--')
        if y_labels:
            axs[n].set_title('label {}'.format(y_labels[n]))
        else:
            axs[n].set_title('label {}'.format(n))
    fig.suptitle('Sample signals by label')
    plt.show()


def print_plot_save_results(arg_dict, history, X, y, channel_names, label_map, train_time, clf_report=None, cm=None,
                            specificity=None):
    """print, plot, and results using keras history object and other arguments"""

    # create results dir
    results_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # save to text file
    current_model = arg_dict['data'] + '_' + arg_dict['model_name'] + '_' + datetime.datetime.now().strftime('%Y-%m-%d @%H.%M.%S')
    result_dir_current_model = os.path.join(results_dir, current_model)
    os.mkdir(result_dir_current_model)
    with open(os.path.join(result_dir_current_model, 'arguments.txt'), 'w') as file:
        file.write(json.dumps(arg_dict))

    with open(os.path.join(result_dir_current_model, 'test_results.txt'), 'w') as file:
        file.write(json.dumps(clf_report))

    print('\n******** RESULTS *************************')
    print('* Data set: {}'.format(arg_dict['data']))
    if 'Livrable' in arg_dict['data']:
        print('* Classification: {}'.format(arg_dict['classification']))
        print('* Channels: {}'.format(arg_dict['channels']))
        print('* Excluded labels: {}'.format(arg_dict['excluded_labels']))
        print('* Excluded positions: {}'.format(arg_dict['excluded_positions']))

    print('* Segment shape: {}'.format(arg_dict['segment_shape']))
    print('* n secs per sample: {} seconds'.format(arg_dict['window_secs']))
    print('* n samples: {0}'.format(len(y)))
    print('* n classes: {0}'.format(len(np.unique(y))))
    for i in range(0, len(np.unique(y))):
        print('* lbl {0} ({1}): {2} ({3:.1f}%)'.format(i, list(label_map.keys())[i], np.count_nonzero(y == i),
                                                       np.count_nonzero(y == i) / len(y) * 100))
    print('* ')
    print('* Model: {}'.format(arg_dict['model_name']))
    if arg_dict['model_type'] == 'signal':
        for key, value in history.params.items():
            print('*    {}: {}'.format(key, value))
    print('* run time: {0:.2f} seconds'.format(train_time))
    print('*')
    print('* training set accuracy:     {0:.1f}%'.format(history.history['accuracy'][-1]*100))
    print('* validation set accuracy:   {0:.1f}%'.format(history.history['val_accuracy'][-1]*100))
    print('* Test set accuracy:         {0:.1f}%'.format(clf_report['accuracy'] * 100))
    print('* Test set F1-score:         {0:.1f}%'.format(clf_report['weighted avg']['f1-score'] * 100))
    print('* Test sensitivity (recall): {0:.1f}%'.format(clf_report['weighted avg']['recall'] * 100))
    print('* Test specificity (FIX THIS):{0:.1f}%'.format(specificity * 100))

    print('*')
    print('* date: {}\n'.format(datetime.datetime.now()))
    print('* results saved to {}'.format(result_dir_current_model))
    print('**************************************************\n')

    if arg_dict['model_type'] == 'signal':
        plt_history = plot_train_val_acc_loss(history)
        plt_history.savefig(os.path.join(result_dir_current_model, 'train_val_acc_loss.png'))

    target_names = label_map.keys()
    plt_cm = plot_confusion_matrix(cm, target_names, title='Test CM: {0} classes, {1} channels,  {2} sec window, accuracy {3:.1f}%'
                                   .format(len(np.unique(y)), arg_dict['channels'], arg_dict['window_secs'], clf_report['accuracy'] * 100))
    plt_cm.savefig(os.path.join(result_dir_current_model, 'test_cm.png'))

    # create plots of strips
    plot_strips(X, y, label_map, channel_names, result_dir_current_model)


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


if __name__ == "__main__":
    # for testing
    from processing.processing import load_keras_sample_time_series_db
    X, y, _ = load_keras_sample_time_series_db()
    #plot_by_label(X, y)
    plot_strips(X, y)
