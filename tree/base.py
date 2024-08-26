"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import os

np.random.seed(42)

class Node:
    def __init__(self, index:int, attribute: str, value: float, class_label: str, next_nodes=[]):
        self.attribute = attribute
        self.value = value
        self.class_label = class_label
        self.index = index
        self.next_nodes = next_nodes
    
    def add_node(self, node):
        self.next_nodes.append(node)
        

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="Entropy", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, x: pd.DataFrame, y: pd.Series, depth=None) -> Node:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        if depth is None:
            depth=self.max_depth
        if(depth == 0):
            return
        features = x.columns
        if check_ifreal(x[features[0]]) and check_ifreal(y):  #riro
            pass
        
        elif check_ifreal(x[features[0]]): #rido
            pass

        elif check_ifreal(y): #diro
            pass

        else:  #dido
            split_attribute = opt_split_attribute(x,y,"entropy",features)
            split_data = split_data_discrete(x,y,split_attribute)
            children = []
            for i in split_data:
                children.append(self.fit(split_data[i][0], split_data[i][1], depth-1))
            node = Node(0, split_attribute, None, None, children)
            return node


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.


        

    def plot(self) -> None:
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
        pass
