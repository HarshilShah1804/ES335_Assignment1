import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
def generate_data(N, M):
    X = np.random.randint(2, size=(N, M))
    y = np.random.randint(2, size=N)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    return X, y

def measure_training_time(model, X, y):
    start_time = time.time()
    model.fit(X, y)
    # model.plot()
    return time.time() - start_time

def measure_prediction_time(model, X):
    start_time = time.time()
    model.predict(X)
    return time.time() - start_time

Ns = [1000, 10000, 100000]
Ms = [10, 100, 1000]


results = {}

for N in Ns:
    training_times = []
    prediction_times = []
    for M in Ms:
        tree = DecisionTree(criterion="information_gain")
        X, y = generate_data(N, M)
        training_time = measure_training_time(tree, X, y)
        training_times.append(training_time)
        prediction_time = measure_prediction_time(tree, X)
        prediction_times.append(prediction_time)
    results[N] = (training_times, prediction_times)

fig, ax = plt.subplots(2, 1)
ax[0].set_title("Training Time")
ax[1].set_title("Prediction Time")
ax[0].set_xlabel("Number of Features")
ax[1].set_xlabel("Number of Features")
ax[0].set_ylabel("Time (s)")
ax[1].set_ylabel("Time (s)")
ax[0].plot(Ms, results[1000][0], label="N=1000")
ax[0].plot(Ms, results[10000][0], label="N=10000")
ax[0].plot(Ms, results[100000][0], label="N=100000")
ax[1].plot(Ms, results[1000][1], label="N=1000")
ax[1].plot(Ms, results[10000][1], label="N=10000")
ax[1].plot(Ms, results[100000][1], label="N=100000")
ax[0].legend()
ax[1].legend()
# plt.tight_layout()
plt.show()