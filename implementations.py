import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def compute_square_loss(y, tx, w):
    """Calculate the square loss

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the square loss (a scalar), corresponding to the input parameters w.
    """
    n = len(y)
    return np.sum((y - tx @ w) ** 2) / (2 * n)


def compute_mse_gradient(y, tx, w):
    """Computes the gradient at w, using the mean squared error distance.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        An numpy array of shape (D,) (same shape as w), containing the gradient of the loss at w.
    """
    n = len(y)
    # error vector
    e = y - tx @ w
    return -(tx.T @ e) / n


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Performs a gradient descent with MSE loss and returns the last computed parameters.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D,). The vector of initial model parameters.
        max_iters: int. The number of iterations of the gradient descent.
        gamma: float. The learning rate.

    Returns:
        A numpy array of shape=(D,), containing the last vector of parameters,
        and a float, equal to the loss value of the model with the last parameters.
    """
    w = initial_w
    loss = compute_square_loss(y, tx, w)

    for n_iter in range(max_iters):
        # update gradient
        w = w - gamma * compute_mse_gradient(y, tx, w)
        # compute loss and print it
        loss = compute_square_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Performs a stochastic gradient descent (with batch size equal to 1) with MSE loss
        and returns the last computed parameters, with the loss.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D,). The vector of initial model parameters.
        max_iters: int. The number of iterations of the gradient descent.
        gamma: float. The learning rate.

    Returns:
        A numpy array of shape=(D,), containing the last vector of parameters,
        and a float, equal to the loss value of the model with the last parameters.
    """
    batch_size = 1  # as specified on project description

    w = initial_w
    loss = compute_square_loss(y, tx, w)

    for n_iter, (minibatch_y, minibatch_tx) in enumerate(
        batch_iter(y, tx, batch_size, max_iters)
    ):
        grad = compute_mse_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma * grad
        loss = compute_square_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Computes the least squares solution and returns the best parameters, together with the loss.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N,D)

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, the loss.
    """
    w = np.linalg.lstsq(tx, y, rcond=None)[0]
    loss = compute_square_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Performs the ridge regression on the given data and returns the best parameters, together with the loss

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, the loss.
    """
    n, d = tx.shape

    left_hand_side = (tx.T @ tx) + (2 * n * lambda_) * np.identity(d)
    right_hand_side = tx.T @ y

    w = np.linalg.lstsq(left_hand_side, right_hand_side, rcond=None)[0]
    loss = compute_square_loss(y, tx, w)
    return w, loss


def sigmoid(t):
    """Apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-t))


def neg_log_likelihood(y, tx, w):
    """Compute the cost of the negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    n = len(y)

    tx_w = tx @ w
    exp_tx_w = np.exp(tx_w)
    cost = -y * tx_w + np.log(1 + exp_tx_w)
    loss = np.sum(cost) / n
    return loss


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    n = len(y)
    return (1 / n) * tx.T @ (sigmoid(tx @ w) - y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Performs a logistic regression using gradient descent

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).
        initial_w: numpy array of shape=(D,). The vector of initial model parameters.
        max_iters: int. The number of iterations of the gradient descent.
        gamma: float. The learning rate.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, the negative log likelihood loss.
    """
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def reg_logistic_regression_one_step(y, tx, lambda_, w, gamma):
    """Do one step of gradient descent using regularized logistic regression, and returns the updated weights.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: scalar.
        w:  shape=(D, 1)
        gamma: float

    Returns:
        w: shape=(D, 1)
    """
    grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    new_w = w - gamma * grad
    return new_w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Performs a regularized logistic regression using gradient descent

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).
        lambda_: scalar.
        initial_w: numpy array of shape=(D,). The vector of initial model parameters.
        max_iters: int. The number of iterations of the gradient descent.
        gamma: float. The learning rate.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, the negative log likelihood loss.
    """
    w = initial_w

    for n_iter in range(max_iters):
        w = reg_logistic_regression_one_step(y, tx, lambda_, w, gamma)

    loss = neg_log_likelihood(y, tx, w)
    return w, loss
