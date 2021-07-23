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

    def fit(self, X, y):
    # m 為資料筆數，n 為特徵數量
    if X.ndim == 1:
        X = X.reshape(X.shape[0], 1)
    m, n = X.shape

    # 是否進行正規化
    if self.feature_scaling:
        X = self.normalize(X)

    # 在 X 左方加入一行 1 對應到參數 theta 0
    X = np.hstack((np.ones((m, 1)), X))

    y = y.reshape(y.shape[0], 1)

    self.W = np.zeros((n+1,1))

    # 每個 iteration 逐步更新參數
    for i in range(self.num_iteration):
        y_hat = X.dot(self.W)
        cost = self.cost_function(y_hat, y, m)
        self.cost_history[i] = cost
        self.gradient_descent(X, y_hat, y, m)

    def normalize(self, X):
    self.M = np.mean(X, axis=0)
    self.S = np.max(X, axis=0) - np.min(X, axis=0)
    return (X - self.M) / self.S

    def cost_function(self, y_hat, y, m):
        return 1/(2*m) * np.sum((y_hat - y)**2)

    def compute_gradient(self, X, y_hat, y, m):
        return 1/m * np.sum((y_hat - y) * X, axis=0).reshape(-1,1)

    def gradient_descent(self, X, y_hat, y, m):
        self.W -= self.learning_rate * self.compute_gradient(X, y_hat, y, m)

    def predict(self, X):
    if X.ndim == 1:
        X = X.reshape(X.shape[0], 1)
    m, n = X.shape

    if self.normalize:
        X = (X - self.M) / self.S

    X = np.hstack((np.ones((m, 1)), X))

    y_hat = X.dot(self.W)
    return y_hat

    def normal_equation(self, X, y):
    if X.ndim == 1:
        X = X.reshape(X.shape[0], 1)
    m, n = X.shape

    X = np.hstack((np.ones((m, 1)), X))

    y = y.reshape(y.shape[0], 1)

    self.W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
