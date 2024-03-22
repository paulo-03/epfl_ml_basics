import numpy as np

from src.helper_visualization import *
from src.model_performance import accuracy, f1_score, split_threshold


def build_poly(x, degree):
    """
    Polynomial basis expansion for input data x, for degree j=0 up to j=degree.

    Args:
        x (np.array): is the number of samples.
        degree (int): the max degree of the polynomial expansion.

    Returns:
        np.array: the polynomial expansion of x.
    """
    n = x.shape[0]
    poly = np.ones((n, 1))
    for deg in range(1, degree + 1):
        poly = np.column_stack((poly, x**deg))

    return poly


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.

    Args:
        y (np.ndarray): the array on which we want to construct the k-fold.
        k_fold (int): K in K-fold, i.e. the number of folds.
        seed (int): the seed used for randomness.

    Returns:
        np.ndarray: the array that indicates the data indices for each fold.
    """
    # How many samples/observations
    rows = len(y)

    # Let's check the length of our future k-folds
    fold_length = int(rows / k_fold)

    # Create a list of indices
    idx = np.arange(rows)

    # Shuffle all that
    np.random.seed(seed)
    np.random.shuffle(idx)

    # Check how many sample we will drop, and drop it if needed (because rows%k-fold != 0)
    del_row = rows % k_fold
    idx = idx[: rows - del_row]

    # Transform in array and reshape it
    idx_shuffle_array = np.array(idx).reshape((k_fold, fold_length))

    return idx_shuffle_array


def compute_best_threshold(
    x_train,
    w_star,
    y_train,
    x_test=None,
    y_test=None,
    number=10,
    range_=None,
    plot=False,
):
    """
    Compute the best threshold from the best f1 score produced by the prediction.
    Print the evolution plot of the metrics for each threshold in the range.

    Args:
        x_train (np.ndarray): The data points used during training.
        w_star (np.ndarray): The best weight vector.
        y_train (np.ndarray): The expected y values used during training.
        x_test (np.ndarray): The data points used for testing.
        y_test (np.ndarray): The data points used fo testing.
        number (int): The number of different values between the range if not specified.
        range_ (np.ndarray): Custom range of threshold to use to compute the best one.
        plot (bool): Boolean telling whether to plot or not the best threshold evolution.

    Returns:
        float: The best threshold.
    """
    y_pred = x_train @ w_star
    if range_ is None:
        range_ = np.linspace(np.min(y_pred), np.max(y_pred), number)
    f1_scores = compute_f1_scores(y_pred, y_train, range_)
    accuracies = compute_accuracies(y_pred, y_train, range_)

    results = np.column_stack((range_, f1_scores, accuracies))
    results = results[np.argmax(f1_scores)]

    if plot:
        if x_test is not None and y_test is not None:
            y_pred_test = x_test @ w_star
            fig, axes = init_subplot(1, 2, (10, 4))
            f1_scores_test = compute_f1_scores(y_pred_test, y_test, range_)
            accuracies_test = compute_accuracies(y_pred_test, y_test, range_)

            results_test = np.column_stack((range_, f1_scores_test, accuracies_test))
            results_test = results_test[np.argmax(f1_scores)]

            # plot the test data
            accuracy_f1_visualization(
                range_,
                accuracies_test,
                f1_scores_test,
                results_test[0],
                axes[1],
                train=False,
            )
            plot_text(
                f"Best model metrics for best f1 score:\n\n"
                f"- Best threshold: {results_test[0]}\n"
                f"- Best f1 score: {results_test[1]}\n"
                f"- Best accuracy: {results_test[2]}",
                axes[1],
            )

            ax = axes[0]
        else:
            fig, ax = init_subplot(1, 1, (6, 5))

        accuracy_f1_visualization(range_, accuracies, f1_scores, results[0], ax)
        plot_text(
            f"Best model metrics for best f1 score:\n\n"
            f"- Best threshold: {results[0]}\n"
            f"- Best f1 score: {results[1]}\n"
            f"- Best accuracy: {results[2]}",
            ax,
        )

        display_plot()

    return results[0]


def compute_f1_scores(y_pred, y, range_):
    """
    Compute the list of f1 scores given a prediction and an expected vector y and a range of thresholds.

    Args:
        y_pred (np.ndarray): The prediction vector.
        y (np.ndarray): The expected vector.
        range_ (np.ndarray): The range of threshold to apply to the prediction.

    Returns:
        np.ndarray: The list of f1 scores
    """
    return [f1_score(split_threshold(y_pred.copy(), t), y) for t in range_]


def compute_accuracies(y_pred, y, range_):
    """
    Compute the list of accuracies given a prediction and an expected vector y and a range of thresholds.

    Args:
        y_pred (np.ndarray): The prediction vector.
        y (np.ndarray): The expected vector.
        range_ (np.ndarray): The range of threshold to apply to the prediction.

    Returns:
        np.ndarray: The list of accuracies
    """
    return [accuracy(split_threshold(y_pred.copy(), t), y) for t in range_]
