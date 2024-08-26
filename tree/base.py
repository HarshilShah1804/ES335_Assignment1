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
from graphviz import Digraph
from IPython.display import Image, display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class Node:
    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=False, output=None, gain=0):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.output = output
        self.gain = gain
    
    def check_leaf(self):
        return self.is_leaf



class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree.

        # If the depth exceeds max_depth or all the target values are the same, create a leaf node
        def build(X: pd.DataFrame, y: pd.Series, depth: int) -> Node:

            # my_criterion, criterion_func = check_criteria(y, self.criterion)
            # criterion_value = criterion_func(y)
            # criterion_pair = (my_criterion, criterion_value)     

            if depth >= self.max_depth or y.nunique() == 1:
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                else:
                    return Node(is_leaf=True, output=y.mode()[0])
            
            best_attribute = opt_split_attribute(X, y, X.columns, self.criterion)

            # If no good split is found, create a leaf node
            if best_attribute is None:
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                else:
                    return Node(is_leaf=True, output=y.mode()[0])

            if check_ifreal(X[best_attribute]):
                best_value = find_optimal_threshold(y, X[best_attribute], self.criterion)
            else:
                best_value = X[best_attribute].mode()[0]

            X_left, y_left, X_right, y_right = split_data(X, y, best_attribute, best_value)

            # If a valid split is not possible, create a leaf node
            if X_left.empty or X_right.empty:
                if check_ifreal(y):
                    return Node(is_leaf=True, output=np.round(y.mean(),4))
                else:
                    return Node(is_leaf=True, output=y.mode()[0])
                
            best_gain = information_gain(y, X[best_attribute], self.criterion)

                
            left_subtree = build(X_left, y_left, depth + 1)
            right_subtree = build(X_right, y_right, depth + 1)

            return Node(attribute=best_attribute, value=best_value, left=left_subtree, right=right_subtree, gain=best_gain)

        self.tree = build(X, y, depth)
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        def predict_row(x: pd.Series) -> float:
            """
            Function to predict the output for a single row of input
            """

            current_node = self.tree
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
            
        return pd.Series([predict_row(x) for _, x in X.iterrows()])


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
        
        if not self.tree:
            print("Tree not trained yet")
            return
        
        dot = Digraph()

        def add_node(node: Node, parent_name: str = None, edge_label: str = None) -> None:
            node_id = str(id(node))
            if node.is_leaf:
                node_label = f"Prediction: {node.output}\n"
            else:
                node_label = f"?(attr {node.attribute} <= {node.value:.2f})\n "
            dot.node(node_id, label=node_label, shape='box' if node.is_leaf else 'ellipse')
            if parent_name:
                dot.edge(parent_name, node_id, label=edge_label)

            if node.left:
                add_node(node.left, node_id, 'Yes')
            if node.right:
                add_node(node.right, node_id, 'No')

        add_node(self.tree)

        print("\nTree Structure:")
        print(self.print_tree())
        # dot.render(path, format="png", view=True, cleanup=True)
        if path:
            dot.render(path, format="png", view=False, cleanup=True)
            display(Image(filename=f"{path}.png"))  
        else:
            png_data = dot.pipe(format='png')
            display(Image(data=png_data))
            
    
    def print_tree(self) -> str:
        def print_node(node: Node, indent: str = '') -> str:
            output = ''
            if node.is_leaf:
                output += f'Class: {node.output}\n'
            else:
                output += f'?(attr {node.attribute} <= {node.value:.2f})\n'
                output += indent + '    Yes: '
                output += print_node(node.left, indent + '    ')
                output += indent + '    No: '
                output += print_node(node.right, indent + '    ')

            return output

        if not self.tree:
            return "Tree not trained yet"
        else:
            return print_node(self.tree)


    def _repr_(self):
        return f"DecisionTree(criterion={self.criterion}, max_depth={self.max_depth})\n\nTree Structure:\n{self.print_tree()}"