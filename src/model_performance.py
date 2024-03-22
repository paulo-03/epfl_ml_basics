import numpy as np

from src.Exceptions import IllegalArgumentException


def split_threshold(y_pred, threshold=0.0):
    """
    Apply the threshold to the y_pred array to return the classification array.

    Args:
        y_pred(np.ndarray): The continuous prediction to transform into a classification prediction.
        threshold(float): The threshold to predict either 1 or -1.

    Returns:
         np.ndarray: The classification predictions.

    """
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = -1
    return y_pred


def predictions(x_test, w, threshold=0.0):
    """
        Computes the predictions, given the test data, the weights, and a threshold.

    Args:
        x_test(np.ndarray): The datapoints we wish to predict.
        w(np.ndarray): The optimal weight.
        threshold(float): the threshold to predict either 1 or -1.
    Returns:
        np.ndarray: predictions.

    """
    y_pred = x_test @ w
    return split_threshold(y_pred, threshold)


def compute_tp_tn_fp_fn(prediction, y):
    """
    Compute the true positive, the true negative, the false positive and the false negative prediction numbers.

    Args:
        prediction (np.ndarray): The prediction array.
        y (np.ndarray): The expected prediction array.

    Returns:
        tuple[int, int, int, int]: The tuple composed of tp, tn, fp, fn.
    """

    # Check arguments same shape
    if prediction.shape != y.shape:
        raise IllegalArgumentException(
            f"Shape of array does not match: {prediction.shape} / {y.shape}"
        )

    # Check arguments same set of elements
    test_pred_in_y = np.isin(prediction, np.unique(y))
    if not np.all(test_pred_in_y):
        diff_vals = prediction[np.invert(test_pred_in_y)]
        raise IllegalArgumentException(
            f"Both arrays should have same elements, "
            f"elemts:{np.unique(diff_vals)} found in prediction array that are not in y array"
        )

    stacked_pred = np.column_stack((prediction, y))
    positive = stacked_pred[stacked_pred[:, 0] == 1]
    negative = stacked_pred[stacked_pred[:, 0] == -1]

    # Number of true positive (model correctly predicted positive class)
    tp = len(positive[positive[:, 1] == 1])
    # Number of false positive (model incorrectly predicted positive class)
    fp = len(positive[positive[:, 1] == -1])
    # Number of true negative (model correctly predicted negative class)
    tn = len(negative[negative[:, 1] == -1])
    # Number of false negative (model incorrectly predicted negative class)
    fn = len(negative[negative[:, 1] == 1])

    return tp, tn, fp, fn


def accuracy(prediction, y):
    """
    Compute the accuracy of the prediction.

    Args:
        prediction (np.ndarray): The prediction array.
        y (np.ndarray): The expected prediction array.

    Returns:
        float: The accuracy of the model.
    """
    tp, tn, fp, fn = compute_tp_tn_fp_fn(prediction, y)

    return (tn + tp) / (tp + tn + fp + fn)


def f1_score(prediction, y):
    """
    Compute the f1 score of the prediction.

    Args:
        prediction (np.ndarray): The prediction array.
        y (np.ndarray): The expected prediction array.

    Returns:
        float: The f1 score of the model.
    """
    tp, tn, fp, fn = compute_tp_tn_fp_fn(prediction, y)

    return tp / (tp + 0.5 * (fp + fn))
