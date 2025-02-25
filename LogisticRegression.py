import numpy as np
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, independent_data, dependent_data):
        count_of_samples, count_of_features = independent_data.shape
        self.weights = np.zeros(count_of_features)
        self.bias = 0

        for i in range(self.max_epochs):
            x = np.dot(independent_data, self.weights) + self.bias
            p = self.obtain_probabilities(x)

            error_values = dependent_data - p

            d_weights = (1/count_of_samples) * np.dot(independent_data.T, error_values)
            d_bias = (1/count_of_samples) * np.sum(error_values)

            self.weights = self.weights - self.regularization * d_weights
            self.bias = self.bias - self.regularization * d_bias

    def predict(self, data_array):
      x = np.dot(data_array, self.weights) + self.bias
      p = self.obtain_probabilities(x)

      return np.where( p > 0.5, 1, 00)

    def obtain_probabilities(self, independent_data):
        return 1 / ( 1 + np.exp( - independent_data))

    def accuracy_score_calc(self, pred, actual):
        if len(pred) != len(actual):
            raise ValueError("Value error")
        currect_count = 0
        for p, a in zip(pred, actual):
            if p == a:
                currect_count += 1
        score = currect_count / len(actual)

        return score