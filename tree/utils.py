"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    one_hot_encoded_data = pd.get_dummies(X, columns = ['Remarks', 'Gender'])
    print(one_hot_encoded_data)

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
    elif criterion == 'gini':
        return gini_index(Y) - sum([(attr == i).sum() / Y.size * gini_index(Y[attr == i]) for i in attr.unique()])
    elif criterion == 'mse':
        return ((Y - attr) ** 2).mean() - sum([(attr == i).sum() / Y.size * ((Y[attr == i] - attr[attr == i]) ** 2).mean() for i in attr.unique()])


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    ig_max = -np.inf
    split_feature = None
    for feature in features:
        ig = information_gain(y, X[feature], criterion)
        if ig > ig_max:
            ig_max = ig
            split_feature = feature
    return split_feature

def split_data_discrete(X: pd.DataFrame, y: pd.Series, attribute):
    """
    Function to get the split of the data for discrete features based on information gain.

    features: pd.Series is a list of all the attributes we have to split upon

    return: split of the dataset based on the optimal attribute
    """
    split_data = {}
    for i in X[attribute].unique():
        split_data[i] = (X[X[attribute] == i], y[X[attribute] == i])
    return  split_data

def split_data_real(X: pd.DataFrame, y: pd.Series, attribute):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
