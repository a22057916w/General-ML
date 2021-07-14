import numpy as np

class LinearRegression():
    def __init__(self, num_iteration=100, learning_rate=1e-1, feature_scaling=True):
        self.num_iteration = num_iteration
        self.learning_rate = learning_rate
        self.feature_scaling = feature_scaling
        self.M = 0 # normalize mean
        self.S = 1 # normalize range
        self.W = None
        self.cost_history = np.empty(num_iteration)
