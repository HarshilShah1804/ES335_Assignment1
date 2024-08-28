import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

# np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5, )

# For plotting
# for class_value in set(y):
#     plt.scatter(X[y == class_value, 0], X[y == class_value, 1], label=f'Class {class_value}')

# Write the code for Q2 a) and b) below. Show your results.
