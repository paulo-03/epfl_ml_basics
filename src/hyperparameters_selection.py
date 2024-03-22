import numpy as np

from implementations import *
from src.helper import build_poly, build_k_indices
from src.helper_visualization import *


def cross_validation(
    y, x, k_indices, k, gamma, degree, regression_func, lambda_=None, max_iters=None
):
    """
    Return the loss of a regression function for a fold corresponding to k_indices.

    Args:
        y (np.ndarray): label.
        x (np.ndarray): data.
        k_indices (np.ndarray): 2D array returned by build_k_indices().
        k (int): the k-th fold (N.B.: not to confused with k_fold which is the fold nums).
        lambda_ (float): cf. ridge_regression().
        degree (int): cf. build_poly().
        gamma (float): learning rate for gradient descent.
        regression_func (str): Regression function to use, can by any of the following strings: <ul><li>
                                - least_square
                                - ridge_regression
                                - reg_logistic_regression
                                - mean_squared_error_gd
                                - mean_squared_error_sgd
        max_iters (int): maximum number of iteration for gradient descent.

    Returns:
        tuple[float, float]: train and test root-mean-square errors (rmse) = sqrt(2*mse)
    """
    # Split data
    x_test = x[k_indices][k]
    y_test = y[k_indices][k]
    x_train = np.delete(x, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k], axis=0)

    # Expand degree of data
    x_train_poly = build_poly(x_train, degree)
    x_test_poly = build_poly(x_test, degree)

    # Compute the initial weight (even if not needed, i.e. least_square method)
    initial_w = np.zeros(x_train_poly.shape[1])

    # Perform the regression
    if regression_func == "least_square":
        w, loss_tr = least_squares(y_train, x_train_poly)
        loss_te = compute_square_loss(y_test, x_test_poly, w)
    elif regression_func == "ridge_regression":
        w, loss_tr = ridge_regression(y_train, x_train_poly, lambda_)
        loss_te = compute_square_loss(y_test, x_test_poly, w)
    elif regression_func == "mean_squared_error_gd":
        w, loss_tr = mean_squared_error_gd(
            y_train, x_train_poly, initial_w, max_iters, gamma
        )
        loss_te = compute_square_loss(y_test, x_test_poly, w)
    elif regression_func == "mean_squared_error_sgd":
        w, loss_tr = mean_squared_error_sgd(
            y_train, x_train_poly, initial_w, max_iters, gamma
        )
        loss_te = compute_square_loss(y_test, x_test_poly, w)
    elif regression_func == "logistic_regression":
        w, loss_tr = logistic_regression(
            y_train, x_train_poly, initial_w, max_iters, gamma
        )
        loss_te = neg_log_likelihood(y_test, x_test_poly, w)
    elif regression_func == "reg_logistic_regression":
        w, loss_tr = reg_logistic_regression(
            y_train, x_train_poly, lambda_, initial_w, max_iters, gamma
        )
        loss_te = neg_log_likelihood(y_test, x_test_poly, w)
    else:
        raise NameError(
            f"Regressions {regression_func} not found.\n"
            f"Please select one of the following regression functions :\n"
            f"- least_square\n"
            f"- ridge_regression\n"
            f"- reg_logistic_regression\n"
            f"- mean_squared_error_gd\n"
            f"- mean_squared_error_sgd "
        )

    return loss_tr, loss_te


def compute_mean_k_fold(
    x, y, k_fold, k_indices, gamma, d, regression_func, lambda_, max_iters
):
    """
     Compute the mean of the k-fold cross validation.

    Args:
        x (np.ndarray): train data.
        y (np.ndarray): train data result.
        k_fold (int): the number of k-fold.
        k_indices (np.ndarray): The list of indices for each k-fold.
        gamma (float): The learning rate.
        d (int): The current degree of the polynomial.
        regression_func (str): The function to use to perform the regression (same as in best_degree_selection)
        lambda_ (float): The current value of lambda parameter for the ridge penalization
        max_iters (int): The maximum number of gradient descent step to perform

    Returns:
        tuple[flaot, float]: The rmse_train and the rmse_test of the k k-fold
    """

    rmse = np.empty((k_fold, 2))
    for k in range(k_fold):
        loss_tr, loss_te = cross_validation(
            y,
            x,
            k_indices,
            k,
            gamma,
            d,
            regression_func,
            lambda_=lambda_,
            max_iters=max_iters,
        )
        rmse[k] = [loss_tr, loss_te]

    [rmse_train, rmse_test] = np.mean(rmse, axis=0)
    return rmse_train, rmse_test


def compute_best_lambda_best_rmse(
    x,
    y,
    k_fold,
    k_indices,
    gamma,
    d,
    regression_func,
    lambdas,
    max_iters,
    axis=None,
    plot_nb_rows=0,
):
    """
    Compute the best lambda along with the best test rmse for a given degree.

    Args:
        x (np.ndarray): train data.
        y (np.ndarray): train data result.
        k_fold (int): the number of k-fold.
        k_indices (np.ndarray): The list of indices for each k-fold.
        gamma (float): The learning rate.
        d (int): The current degree of the polynomial.
        regression_func (str): The function to use to perform the regression (same as in best_degree_selection).
        lambdas (np.ndarray): The list of lambda value to test.
        max_iters (int): The maximum number of gradient descent step to perform.
        axis (plt.Axes): The axis on which to draw the cross validation visu.
        plot_nb_rows (int): The number of rows in the subplots

    Returns:
        tuple[float, float]: The best lambda value for the given degree along with the best test rmse value.
    """
    # Initialize list to store losses
    rmse_te = []
    rmse_tr = []

    # Go through all possible lambdas
    for lambda_ in lambdas:
        # Compute the mean rmse loss for specific degree and lambda
        rmse_train, rmse_test = compute_mean_k_fold(
            x, y, k_fold, k_indices, gamma, d, regression_func, lambda_, max_iters
        )

        # Store values to be able to find the best lambda
        rmse_tr.append(rmse_train)
        rmse_te.append(rmse_test)

    # Select the best rmse and return its associate lambda
    min_val_d = np.argmin(rmse_te)

    # print evolution of training for each degree
    print(
        f"Cross validation k-fold results:\n"
        f"====================\n"
        f"\t- Degree -> {d}\n"
        f"\t- Best lambda -> {lambdas[min_val_d]}\n"
        f"\t- Best rmse_te -> {rmse_te[min_val_d]}"
    )

    # print the loss evolution of train and test set for lambdas for current degree
    if axis is not None:
        # The way to retrieve the axis to draw depends on the shape of the subplots
        if type(axis) is not np.ndarray:
            ax_id = axis
        else:
            ax_id = (
                axis[d - 1]
                if plot_nb_rows == 1
                else axis[int((d - 1) / 4), (d - 1) % 4]
            )
        cross_validation_visualization(lambdas, rmse_tr, rmse_te, ax_id, d)

    return lambdas[min_val_d], rmse_te[min_val_d]


def best_degree_selection(
    x, y, degrees, k_fold, gamma, regression_func, seed, lambdas=None, max_iters=None
):
    """
    K-fold cross validation over regularisation parameter lambda and degree.

    Args:
        x (np.ndarray): The array datapoints.
        y (np.ndarray): The array of expected predictions.
        degrees (np.ndarray): The list of all degrees to test.
        k_fold (int): The number of folds.
        lambdas (np.ndarray or None): The array of lambdas to test.
        regression_func (str):    Regression function to use, can by any of the following strings:
                                - least_square
                                - ridge_regression
                                - reg_logistic_regression
                                - mean_squared_error_gd
                                - mean_squared_error_sgd
        seed (int): fix the randomness for reproducibility reasons.
        gamma (float): learning rate for gradient descent.
        max_iters (int): maximum number of iteration for gradient descent
    Returns:
        tuple[float, float, float]:
            best_degree : integer, value of the best degree
            best_lambda : scalar, value of the best lambda
            best_rmse : value of the rmse for the couple (best_degree, best_lambda)
    """

    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # Initialize the list to store best lambdas and rmses for each degree
    best_lambdas = []
    best_rmses = []

    if lambdas is None:
        lambdas = [0]
        ax = None
        rows = 0
    else:
        rows = int((len(degrees) - 1) / 4) + 1
        col = len(degrees) if rows == 1 else 4
        fig, ax = init_subplot(rows, col, (4 * col, 3 * rows), sharey=True)

    # Start looking for best hyperparameters
    for d in degrees:
        best_lambda, best_rmse = compute_best_lambda_best_rmse(
            x,
            y,
            k_fold,
            k_indices,
            gamma,
            d,
            regression_func,
            lambdas,
            max_iters,
            ax,
            plot_nb_rows=rows,
        )

        best_lambdas.append(
            best_lambda
        )  # pair the best lambda with its current degree d
        best_rmses.append(
            best_rmse
        )  # store the rmse loss of the pair (deg & best lambda)

    if ax is not None:
        display_plot()

    # Select the best rmse and return its associate pair (deg & best lambda)
    min_val = np.argmin(best_rmses)
    best_lambda = best_lambdas[min_val]
    best_degree = degrees[min_val]
    best_rmse = best_rmses[min_val]

    return best_degree, best_lambda, best_rmse


def split_data(x, y, ratio, seed):
    """
    Split the dataset based on the split ratio. If ratio is 0.8 you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round you can use np.floor. Also
    check the documentation for np.random.permutation, it could be useful.

    Args:
        x (np.ndarray): data.
        y (np.ndarray): labels.
        ratio (float): scalar in [0,1].
        seed (int): seed of randomness.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            x_tr: train data.
            x_te: test data.
            y_tr: train labels.
            y_te: test labels.
    """
    # Set random seed
    np.random.seed(seed)

    # Create the randomizer and perform the
    n = x.shape[0]
    randomizer = np.random.permutation(n)
    r_x = x[randomizer]
    r_y = y[randomizer]

    train_n = int(n * ratio)
    return r_x[:train_n], r_x[train_n:], r_y[:train_n], r_y[train_n:]
