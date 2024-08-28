"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Union
from IPython.display import Image, display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class Node:
    # The attribute to split upon for non-leaf nodes, None for leaf nodes, and the output value for leaf nodes, gain is the information gain of the split, is_leaf is a boolean value to check if the node is a leaf node, value is the threshold value for real attributes, left and right are the left and right child nodes.
    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=False, output=None, gain=0):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.output = output
        self.gain = gain
    
    # Function to check if the node is a leaf node
    def check_leaf(self):
        return self.is_leaf



class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root_node = None


    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree.

        # If the depth exceeds max_depth or all the target values are the same, create a leaf node
        def build(X: pd.DataFrame, y: pd.Series, depth: int) -> Node:  
            #checking if maximum depth is reached or all the target values are same
            if depth >= self.max_depth or y.nunique() == 1:
                # If the target values are real, return the mean of the target values
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                # If the target values are discrete, return the mode of the target values
                else:
                    return Node(is_leaf=True, output=y.mode()[0])
            
            # Find the best attribute to split upon
            best_attribute = opt_split_attribute(X, y, X.columns, self.criterion)

            # If no good split is found, create a leaf node
            if best_attribute is None:
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                else:
                    return Node(is_leaf=True, output=y.mode()[0])

            if check_ifreal(X[best_attribute]):
                opt_val = opt_threshold(y, X[best_attribute], self.criterion)
            else:
                opt_val = X[best_attribute].mode()[0]

            # Split the data based on the best attribute and value
            X_left, y_left, X_right, y_right = split_data(X, y, best_attribute, opt_val)

            # If a valid split is not possible, create a leaf node
            if X_left.empty or X_right.empty:
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                else:
                    return Node(is_leaf=True, output=y.mode()[0])
                
            best_gain = information_gain(y, X[best_attribute], self.criterion)

            # Recursively build the left and right subtrees
            left = build(X_left, y_left, depth + 1)
            right = build(X_right, y_right, depth + 1)

            return Node(attribute=best_attribute, value=opt_val, left=left, right=right, gain=best_gain)

        # Start building the tree
        self.root_node = build(X, y, depth)
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        def predict_single(x: pd.Series) -> float:
            """
            Function to predict the output for a single row of input
            """

            current_node = self.root_node
            while not current_node.check_leaf():
                if check_ifreal(x[current_node.attribute]):
                    if x[current_node.attribute] <= current_node.value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
                else:
                    if x[current_node.attribute] == current_node.value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
            return current_node.output
            
        return pd.Series([predict_single(x) for _, x in X.iterrows()])


    def plot(self, path=None) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        
        if not self.root_node:
            print("Tree not trained yet")
            return

        print("\nTree Structure:")
        print(self.print_tree())
            
    
    def print_tree(self) -> str:
        def print_node(node: Node, indent: str = '') -> str:
            if node.is_leaf:
                return f'Class {node.output}\n'
            
            # Format the non-leaf node
            result = f'?(attribute {node.attribute} <= {node.value:.2f})\n'
            result += f'{indent}Y: {print_node(node.left, indent + "    ")}'
            result += f'{indent}N: {print_node(node.right, indent + "    ")}'
            
            return result

        if not self.root_node:
            return "Tree not trained yet"
        else:
            return print_node(self.root_node)


    