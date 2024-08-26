"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from typing import Union

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    pass



def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    assert y.size > 0
    return isinstance(y[0], (int, float))


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    assert Y.size > 0
    entropy = 0.
    for i in Y.unique():
        p = (Y == i).sum() / Y.size
        entropy += -p * np.log2(p)
    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    assert Y.size > 0
    gini_index = 1.
    for i in Y.unique():
        p = (Y == i).sum() / Y.size
        gini_index -= p ** 2
    return gini_index


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    assert Y.size == attr.size
    if criterion == 'entropy':
        return entropy(Y) - sum([(attr == i).sum() / Y.size * entropy(Y[attr == i]) for i in attr.unique()])
    elif criterion == 'gini_index':
        return gini_index(Y) - sum([(attr == i).sum() / Y.size * gini_index(Y[attr == i]) for i in attr.unique()])
    elif criterion == 'mse':
        return ((Y - attr) ** 2).mean() - sum([(attr == i).sum() / Y.size * ((Y[attr == i] - attr[attr == i]) ** 2).mean() for i in attr.unique()])


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features: pd.Series, criterion):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_attribute = None
    best_gain = -float('inf')

    for attribute in features:
        X_col = X[attribute]
        
        # Check if the attribute is real or discrete
        if check_ifreal(X_col):
            threshold = find_optimal_threshold(y, X_col, criterion)
            if threshold is None:
                continue
            
            gain = information_gain(y, X_col <= threshold, criterion) if criterion == "information_gain" else gini_index(y)
        else:
            # For discrete attributes, just calculate the criterion directly
            gain = information_gain(y, X_col, criterion) if criterion == "information_gain" else gini_index(y)

        # Update the best attribute if the current gain is better
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    return best_attribute

def find_optimal_threshold(y: pd.Series, X_col: pd.Series, criterion: str) -> float:
    best_threshold = None
    best_gain = -float('inf')

    # Sort the values and corresponding target labels
    sorted_indices = X_col.argsort()
    X_sorted = X_col.iloc[sorted_indices]
    y_sorted = y.iloc[sorted_indices]

    # Iterate through all possible thresholds
    for i in range(1, len(y_sorted)):
        if X_sorted.iloc[i] == X_sorted.iloc[i-1]:
            continue
        
        # Define the threshold as the midpoint between adjacent values
        threshold = (X_sorted.iloc[i] + X_sorted.iloc[i-1]) / 2
        
        # Split the data based on the threshold
        left_mask = X_sorted <= threshold
        right_mask = X_sorted > threshold
        
        y_left = y_sorted[left_mask]
        y_right = y_sorted[right_mask]

        # Compute the criterion (e.g., information gain)
        if criterion == "information_gain":
            gain = information_gain(y_sorted, X_sorted, criterion)
        elif criterion == "gini_index":
            gain = gini_index(y_sorted, X_sorted)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # Update the best threshold if the current gain is better
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold


def split_data_discrete(X: pd.DataFrame, y: pd.Series, attribute, threshold=0):
    """
    Function to get the split of the data for discrete features based on information gain.

    features: pd.Series is a list of all the attributes we have to split upon

    return: split of the dataset based on the optimal attribute
    """
    left_mask = X[attribute] == X[attribute].mode()[0]  # Splitting based on the most frequent category
    right_mask = ~left_mask

    left_X = X[left_mask]
    right_X = X[right_mask]
    left_y = y[left_mask]
    right_y = y[right_mask]

    return left_X, right_X, left_y, right_y
    

def split_data_real(X: pd.DataFrame, y: pd.Series, attribute, threshold):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    left_mask = X[attribute] <= threshold
    right_mask = ~left_mask

    left_X = X[left_mask]
    right_X = X[right_mask]
    left_y = y[left_mask]
    right_y = y[right_mask]

    return left_X, right_X, left_y, right_y

def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value: Union[float, int, str]) -> tuple:
    """
    Splits the dataset into two based on the given attribute and value.

    Parameters:
    - X (pd.DataFrame): The feature dataset.
    - y (pd.Series): The target labels.
    - attribute (str): The attribute/column name to split on.
    - value (Union[float, int, str]): The value to split the attribute on.

    Returns:
    - X_left (pd.DataFrame): Subset of X where the attribute's value is <= value (for continuous) or == value (for discrete).
    - y_left (pd.Series): Corresponding subset of y for X_left.
    - X_right (pd.DataFrame): Subset of X where the attribute's value is > value (for continuous) or != value (for discrete).
    - y_right (pd.Series): Corresponding subset of y for X_right.
    """
    if check_ifreal(X[attribute]):
        # For continuous attributes, split based on the threshold
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
    else:
        # For discrete attributes, split based on equality
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    return X_left, y_left, X_right, y_right
