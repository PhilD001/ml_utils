import numpy as np


def sensitivity_given_specificity(fpr, tpr, thresholds, desired_specificity=0.9, verbose=False):
    """ compute sensitivity given a required specificity

    Arguments:
        fpr: ndarray. Array of false positive rates
        tpr: ndarray. Array of true positive rates
        Thresholds: ndarray. Array of possible threshold values
        desired_specificity: float: The specificity we are trying to obtain
        verbose: bool. If True, results are printer to screen

    Returns:
         associated_sensitivity: float. The sensitivity given the required specificity

    Notes: fpr, tpr, thesholds can be obtained via  from sklearn.metrics.roc_curve
    e.g.:  fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
    """

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

