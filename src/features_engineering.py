import numpy as np


def risk_scoring(column, y):
    """
    For a specific column, compute the mapping for its categorical values to risk scores.

    Args:
        column (np.ndarray): data vector of a specific feature.
        y (np.ndarray): health situation of each sample.

    Returns:
        dict: mapping dictionary from value (key) to risk.
    """
    # List all different categories
    values = np.unique(column)

    # Dictionary to map values to risk score for specific column
    risk = {}
    for value in values:
        index_samples_with_value = np.where(column == value)[0]
        ill_samples = (y[index_samples_with_value] == 1).sum()
        frac_ill_samples = ill_samples / len(index_samples_with_value)
        risk[value] = frac_ill_samples

    return risk


def risk_mapping(x, y):
    """
    For all columns/features, compute the mapping for their categorical values to risk scores.

    Args:
        x (np.ndarray): data array.
        y (np.ndarray): health situation of each sample.

    Returns:
        list[dict]: list of dictionary for all features mapping value (key) to risk.
    """
    attributes = x.shape[1]
    risk_map = []
    for attribute in range(attributes):
        # Select the column
        column = x[:, attribute]
        # Compute the risk mapping of the current attribute
        risk_dict = risk_scoring(column, y)
        # Append the risk dictionary to the mapping list
        risk_map.append(risk_dict)

    return risk_map


def risk_transform(x, map):
    """
    Transform a dataset from categorical values to risk scores.

    Args:
        x (np.ndarray): data set with categorical values.
        map (list[dict]): list of dictionaries with mapping from values to risk scores.

    Returns:
        np.ndarray: numpy array of data mapped from value to risk.
    """

    # Create an empty array to store the mapped values
    mapped_data = np.empty(x.shape, dtype=np.float64)
    attributes = x.shape[1]

    # Iterate through each column and map values using the corresponding dictionary
    for attribute in range(attributes):
        column = x[:, attribute]
        dictionary = map[attribute]
        mapped_column = np.array([dictionary.get(value, 0) for value in column])
        mapped_data[:, attribute] = mapped_column

    return mapped_data


def standardize(x_train, x_test):
    """
    Standardize the train and test data.

    Args:
        x_train (np.ndarray): train data.
        x_test (np.ndarray): test data.

    Returns:
        tuple[np.ndarray, np.ndarray]: the normalized train and test data.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)

    return (x_train - mean) / std, (x_test - mean) / std


def features_information(x, y):
    """Give us an insight and statistical metrics of each given attribute.

    Args:
        x (np.ndarray): array containing all data.
        y (np.ndarray): array containing label.

    Returns:
        tuple[list[int], list[float]]: the list with the number of different unique values of each attribute
        and the list with the Pearson correlation coefficient of each attribute.
    """
    # Number of different attributes
    num_attributes = x.shape[1]

    # Set the dictionary that will store all information
    count_unique_values_list = []
    correlation_coef_list = []

    # Let's run all the attributes and gain an insight
    for attribute in range(num_attributes):
        column = x[:, attribute]
        count_unique_values = len(np.unique(column))
        correlation_coef = np.corrcoef(column, y)[0, 1]

        # Store the gained information
        count_unique_values_list.append(count_unique_values)
        correlation_coef_list.append(correlation_coef)

    return count_unique_values_list, correlation_coef_list


def features_analyze(x, y, max_unique_values, min_correlation_coef):
    """
    Analyze the features of the dataset. Calculate the number of unique values per features, calculate the correlation
    between the features and the expected output.

    Apply some threshold to the calculated list to perform a filter, print and return them.

    Args:
        x (np.ndarray): The dataset.
        y (np.ndarray): The expected value for each entry of the dataset.
        max_unique_values (int): The threshold for the maximum unique values to still consider a feature.
        min_correlation_coef (float): The threshold for the minimum correlation of a feature with the output to still
                              consider a feature.

    Returns: The indices of the features with less unique values than the threshold, the indices of the features with
             higher correlation than the threshold, the intersection of both indices list.
    """
    count_unique_values, correlation_coef = features_information(x, y)

    # Features' index with few different unique values
    categorical_values_index = np.where(
        np.array(count_unique_values) < max_unique_values
    )[0].tolist()

    # Features' index with bigger correlation coefficient than selected
    correlated_values_index = np.where(
        np.array(correlation_coef) > min_correlation_coef
    )[0].tolist()

    # Features that respect both constraint
    both_values_index = np.intersect1d(
        categorical_values_index, correlated_values_index
    ).tolist()

    # Print the interesting index
    print(
        f"Features with less than {max_unique_values} unique values : {len(categorical_values_index)}\n"
        f"\t{categorical_values_index}\n\n"
        f"Features with more than {min_correlation_coef} of correlation coefficient : {len(correlated_values_index)}\n"
        f"\t{correlated_values_index}\n\n"
        f"Features respecting both constraints : "
        f"{len(both_values_index)}\n"
        f"\t{both_values_index}"
    )

    return categorical_values_index, correlated_values_index, both_values_index


def over_sample(x_train, y_train, seed):
    """
    Oversample the input samples to balance the classes -1 (or 0) and +1.

    Args:
        x_train (np.ndarray): train input.
        y_train (np.ndarray): train label.
        seed (int): the seed used for randomness.

    Returns:
        tuple[np.ndarray, np.ndarray]: The oversampled data input and label.
    """
    # Concatenate the input with the output
    y_train_reshaped = y_train.reshape((len(y_train), 1))
    x_y = np.hstack((x_train, y_train_reshaped))

    # Split the data in negative and positive samples
    x_y_negative = x_y[y_train <= 0, :]
    x_y_positive = x_y[y_train == 1, :]

    # Compute the number of missing positive samples
    number_positive = x_y_positive.shape[0]
    number_negative = x_y_negative.shape[0]
    missing_positive = number_negative - number_positive

    # Safety check, in case the dataset changes
    if missing_positive <= 0:
        return x_train, y_train

    # Sample #missing positive samples (with replacement of course)
    np.random.seed(seed)
    indx = np.arange(number_positive)
    indxs = np.random.choice(indx, missing_positive)
    x_y_additional_positives = x_y_positive[indxs, :]

    # Stack the sampled data with the original data
    x_y_oversampled = np.vstack((x_y, x_y_additional_positives))

    # Split back the data into input (x) and output (y)
    x_oversampled, y_oversampled = np.hsplit(
        x_y_oversampled, np.array([x_y_oversampled.shape[1] - 1])
    )

    return x_oversampled.copy(), y_oversampled[:, 0].copy()


def under_sample(x_train, y_train, seed):
    """
    Deletes random negative samples to get a balanced dataset.

    Args:
        x_train (np.ndarray): train input.
        y_train (np.ndarray): train label.
        seed (int): the seed used for randomness.

    Returns:
        tuple[np.ndarray, np.ndarray]: The undersampled data input and label.
    """
    # Compute the number of missing positive samples
    number_negative = len(y_train[y_train <= 0])
    number_positive = len(y_train[y_train == 1])
    missing_positive = number_negative - number_positive

    # Sanity check, in case the dataset changes
    if missing_positive <= 0:
        return x_train, y_train

    # Computes the negative indices
    np.random.seed(seed)
    negative_indices = np.where(y_train <= 0)[0]

    # Remove the negative samples to get a balanced dataset
    indices_to_remove = np.random.choice(
        negative_indices, missing_positive, replace=False
    )
    x_undersampled = np.delete(x_train, indices_to_remove, axis=0)
    y_undersampled = np.delete(y_train, indices_to_remove, axis=0)

    return x_undersampled, y_undersampled
