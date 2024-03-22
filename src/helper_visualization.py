import matplotlib.pyplot as plt


def cross_validation_visualization(lambdas, rmse_tr, rmse_te, ax, degree):
    """
    Helper function to visualize the evolution of the test rmse and the train rmse in function of the lambda parameters.

    Args:
        lambdas (numpy.ndarray): The list of lambda parameters used in the cross validation.
        rmse_tr (list): The list of rmse for the training samples.
        rmse_te (list): The list of rmse for the testing samples.
        ax (plt.Axes): The axes on which to plot the data.
        degree (int): The degree for which we are currently plotting the rmse evolution.
    """
    ax.plot(lambdas, rmse_tr, marker=".", color="b", label="train error")
    ax.plot(lambdas, rmse_te, marker=".", color="r", label="test error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("r mse")
    ax.set_title(f"cross validation degree {degree}")
    ax.legend(loc=2)
    ax.grid(True)


def init_subplot(row, col, size, sharey=False):
    """
    Helper function to initiate a subplot.

    Args:
        row (int): The number of rows in the subplot.
        col (int): The number of columns in the subplot.
        size (tuple[int, int]): The size of the figure
        sharey (bool): Whether to share the y-axis or not

    Returns:
        tuple[plt.Figure, numpy.ndarray]: The newly created subplots

    """
    return plt.subplots(row, col, figsize=size, layout="constrained", sharey=sharey)


def accuracy_f1_visualization(
    thresholds, accuracies, f1s, best_threshold, ax, train=True
):
    """
    Helper function to see the evolution of the accuracy and f1 score,
    based on the thresholds

    Args:
        thresholds (numpy.ndarray): The list of threshold used to calculate both metrics
        accuracies (list): The list of results of accuracy for each threshold
        f1s (list): The list of f1 scores for each threshold
        best_threshold (float): The best threshold among all computed
        ax (plt.Axes): The axis to which to add the plot
        train(bool): Boolean telling whether this plot display the training or testing data
                     (i.e. true for training, false for testing)
    """

    ax.plot(thresholds, accuracies, marker=".", color="b", label="Accuracy")
    ax.plot(thresholds, f1s, marker=".", color="r", label="F1")
    ax.axvline(best_threshold, ls="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Measure results")
    ax.set_title(
        "Accuracy and f1-score for each threshold "
        + ("Training" if train else "Testing")
        + " data"
    )
    ax.legend(loc=1)
    ax.grid(True)


def plot_text(text, ax, xy=(0.5, -140)):
    """
    Helper function to display a text under graph.

    Args:
        text (str): The text to display
        ax (plt.Axes): The axis on which to display the string.
        xy (tuple[float, float]): The position of the text on the graph, x in 'axes fraction', and y in 'axes pixels'.
    """
    ax.annotate(
        text,
        xy=xy,
        xytext=(0, 0),
        xycoords=("axes fraction", "axes pixels"),
        textcoords="offset points",
        size=11,
        ha="center",
        va="bottom",
    )


def display_plot():
    """
    Helper function to show the plot
    """
    plt.show()
