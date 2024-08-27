import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

#here our target is to predict the mpg of the car
#we will use the following features to predict the mpg of the car
#1. cylinders
#2. displacement
#3. horsepower
#4. weight
#5. acceleration

#model year, origin and car name are not useful for our prediction
X = data[["cylinders", "displacement", "horsepower", "weight", "acceleration"]]
y = data["mpg"]

#horsepower contains some junk values, we will replace them with the mean of the column
#first we will convert the junk values to NaN
X["horsepower"] = pd.to_numeric(X["horsepower"], errors='coerce')
#replace NaN with the mean of the column
X["horsepower"] = X["horsepower"].fillna(X["horsepower"].mean())

#split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
tree.plot()
print("Our Decision Tree")
print("RMSE:", rmse(y_hat, y_test))
print("MAE:", mae(y_hat, y_test))

#using the decision tree module from scikit learn
sklearn_tree = DecisionTreeRegressor(max_depth=5)
sklearn_tree.fit(X_train, y_train)
y_hat = sklearn_tree.predict(X_test)
plot_tree(sklearn_tree)
print("Scikit Learn Decision Tree")
print("RMSE:", rmse(y_hat, y_test))
print("MAE:", mae(y_hat, y_test))
plt.show()
