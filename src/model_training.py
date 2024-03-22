from implementations import *
from src.helper import build_k_indices
from src.model_performance import predictions, f1_score, accuracy


def fit_model(
    x,
    y,
    regression_func,
    k_fold,
    seed,
    gamma=0.7,
    lambda_=0,
    max_iters=10,
    plot_best_threshold=False,
):
    """

    Args:
        x:
        y:
        regression_func:
        k_fold:
        gamma:
        lambda_:
        seed:
        max_iters:
        plot_best_threshold: Plot the best_threshold for training and test data for each k_fold

    Returns:

    """
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # Initialize the list to store weights and performance for each k iteration
    weights = []
    accuracies = []
    f1_scores = []

    for k in range(k_fold):
        w, accuracy, f1_score = cross_training(
            y,
            x,
            k_indices,
            k,
            gamma,
            regression_func,
            lambda_=lambda_,
            max_iters=max_iters,
            plot_best_threshold=plot_best_threshold,
        )

        weights.append(w)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        print(f"Fold {k} finished !")

    mean_w = np.mean(weights, axis=0)

    return mean_w, accuracies, f1_scores


def cross_training(
    y, x, k_indices, k, gamma, regression_func, lambda_, max_iters, plot_best_threshold
):
    """return the weight and performances metrics of a regression function for a fold corresponding to k_indices

    Args:
        y: np.ndarray       shape=(N,)
        x:                  shape=(N,)
        k_indices:          2D array returned by build_k_indices()
        k:                  scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:            scalar, cf. ridge_regression()
        gamma:              learning rate for gradient descent
        regression_func:    Regression function to use, can by any of the following strings: <ul><li>
                                - least_square
                                - ridge_regression
                                - reg_logistic_regression
                                - mean_squared_error_gd
                                - mean_squared_error_sgd
        max_iters:          maximum number of iteration for gradient descent

    Returns:
        mean_w:
        accuracy:
        f1_score:
    """
    # Split data with k to be the test set
    x_test = x[k_indices][k]
    y_test = y[k_indices][k]
    x_train = np.delete(x, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k], axis=0)

    # Compute the initial weight (even if not needed, i.e. least_square method)
    initial_w = np.ones(x_train.shape[1])

    # Perform the regression
    if regression_func == "least_square":
        w, _ = least_squares(y_train, x_train)
    elif regression_func == "ridge_regression":
        w, loss_tr = ridge_regression(y_train, x_train, lambda_)
    elif regression_func == "mean_squared_error_gd":
        w, loss_tr = mean_squared_error_gd(
            y_train, x_train, initial_w, max_iters, gamma
        )
    elif regression_func == "mean_squared_error_sgd":
        w, loss_tr = mean_squared_error_sgd(
            y_train, x_train, initial_w, max_iters, gamma
        )
    elif regression_func == "logistic_regression":
        w, loss_tr = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
    elif regression_func == "reg_logistic_regression":
        w, loss_tr = reg_logistic_regression(
            y_train, x_train, lambda_, initial_w, max_iters, gamma
        )
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

    # Reset the output as initial
    y_train_neg = y_train.copy()
    y_train_neg[y_train_neg == 0] = -1  # reset 0 to -1
    y_test_neg = y_test.copy()
    y_test_neg[y_test_neg == 0] = -1  # reset 0 to -1

    # Compute the performance metrics
    best_threshold = compute_best_threshold(
        x_train, w, y_train_neg, x_test, y_test_neg, number=30, plot=plot_best_threshold
    )
    pred = predictions(x_test, w, best_threshold)
    accuracy_k = accuracy(pred, y_test_neg)
    f1_score_k = f1_score(pred, y_test_neg)

    return w, accuracy_k, f1_score_k
