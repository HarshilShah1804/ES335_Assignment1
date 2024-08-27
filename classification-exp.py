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
m, n = X.shape

## Q2 a)
train_size = 0.7
X_train, X_test = X[:int(m * train_size)], X[int(m * train_size):]
y_train, y_test = y[:int(m * train_size)], y[int(m * train_size):]

tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(pd.DataFrame(X_train), pd.Series(y_train))
y_hat = tree.predict(pd.DataFrame(X_test))
tree.plot()
print("Criteria :", tree.criterion)
print("Accuracy: ", accuracy(pd.Series(y_hat), pd.Series(y_test)))
for cls in np.unique(y):
        print(cls)
        print("Precision: ", precision(y_hat, y_test, cls))
        print("Recall: ", recall(y_hat, y_test, cls))


plt.legend()
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

## Q2 b)
for i in range(5):
    start_test = i * 0.2
    end_test = start_test + 0.2
    
    # Create train and test splits
    X_train = np.concatenate((X[:int(start_test * m)], X[int(end_test * m):]))
    X_test = X[int(start_test * m):int(end_test * m)]
    y_train = np.concatenate((y[:int(start_test * m)], y[int(end_test * m):]))
    y_test = y[int(start_test * m):int(end_test * m)]
    
    print(X_train.shape)
    accuracies = []
    
    for depth in range(2, 8):
        tree = DecisionTree(criterion="information_gain", max_depth=depth)
        tree.fit(pd.DataFrame(X_train), pd.Series(y_train))
        y_hat = tree.predict(pd.DataFrame(X_test))
        
        print("Criteria: information_gain")
        print("Depth: ", depth)
        
        acc = accuracy(pd.Series(y_hat), pd.Series(y_test))
        accuracies.append(acc)
        
        print("Accuracy: ", acc)
        print("\n")
    
    # Plot accuracies for this fold
    plt.plot(range(2, 8), accuracies, label=f"Test {i+1}")

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Tree Depth for 5-Fold Cross-Validation")
plt.legend()
plt.show()

