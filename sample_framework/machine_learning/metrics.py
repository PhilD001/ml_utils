import numpy as np
from sklearn.metrics._base import _average_binary_score
from sklearn.metrics._ranking import _binary_roc_auc_score, _multiclass_roc_auc_score
from sklearn.utils.multiclass import type_of_target, check_array
from sklearn.preprocessing import label_binarize, OneHotEncoder
from functools import partial


def my_roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None, max_fpr=None, multi_class="ovr",
                     labels=None,):
    """same as roc_auc_score from sklearn.metrics.roc_auc_score except:
     - multi_class set to 'ovr
     - y_true and y_score labels are onehot encoded"""

    onehot_encoder = OneHotEncoder(sparse=False)
    y_true = y_true.reshape(len(y_true), 1)
    y_true = onehot_encoder.fit_transform(y_true)

    y_score = y_score.reshape(len(y_score), 1)
    y_score = onehot_encoder.fit_transform(y_score)

    y_type = type_of_target(y_true, input_name="y_true")
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)

    if y_type == "multiclass" or (
        y_type == "binary" and y_score.ndim == 2 and y_score.shape[1] > 2
    ):
        # do not support partial ROC computation for multiclass
        if max_fpr is not None and max_fpr != 1.0:
            raise ValueError(
                "Partial AUC computation not available in "
                "multiclass setting, 'max_fpr' must be"
                " set to `None`, received `max_fpr={0}` "
                "instead".format(max_fpr)
            )
        if multi_class == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        return _multiclass_roc_auc_score(
            y_true, y_score, labels, multi_class, average, sample_weight
        )
    elif y_type == "binary":
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, classes=labels)[:, 0]
        return _average_binary_score(
            partial(_binary_roc_auc_score, max_fpr=max_fpr),
            y_true,
            y_score,
            average,
            sample_weight=sample_weight,
        )
    else:  # multilabel-indicator
        return _average_binary_score(
            partial(_binary_roc_auc_score, max_fpr=max_fpr),
            y_true,
            y_score,
            average,
            sample_weight=sample_weight,
        )


