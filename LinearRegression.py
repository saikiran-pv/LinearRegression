import numpy as np
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0.01, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        best_val_loss = float('inf')
        number_of_samples, number_of_features = X.shape

        if len(y.shape) > 1:
          # Initializing the two dimensional array for weights while predicting the multiple outputs
          sample_count, output_features_count = y.shape
          self.weights = np.zeros((number_of_features, output_features_count))
        else:
            #incase of the single output feature initializing with one dimensional array
          self.weights = np.zeros(number_of_features)

        #setting initial bias to zero
        self.bias = 0

        # initializzing the empty array to store the loss values
        loss = []
        # initializing the ideal weights to zeros
        ideal_weights = 0
        ideal_bias = 0
        # splitting the data into 10% and 90% for testing and training respectively
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        #initializing the patience count to zero to make early stopping when there is no improvement
        patience_count = 0

        #iterating over the allowed number of iterations to find the best params
        for _ in range(self.max_epochs):
            loss_sum_for_batch = []

            # iterating as the batches using the provided batch size
            for b in range(0, number_of_samples, self.batch_size):
                X_train_batch = X_train[b:b+batch_size]
                y_train_batch = y_train[b:b+batch_size]
                
                y_predicted = self.predict(X_train_batch)
                error_values = y_predicted - y_train_batch

                dweights = (1 / number_of_samples) * np.dot(X_train_batch.T, error_values)
                dbias = (1 / number_of_samples) * np.sum(error_values)

                self.weights = self.weights - self.regularization * dweights
                self.bias = self.bias - self.regularization * dbias

                val_loss = self.score(X_val, y_val)

                loss_sum_for_batch.append(val_loss)

            loss_for_batch = np.array(loss_sum_for_batch).mean()
            loss.append(loss_for_batch)
            if loss_for_batch < best_val_loss:
                ideal_weights = np.copy(self.weights)
                ideal_bias = self.bias
                best_val_loss = loss_for_batch
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print("Validation loss is not improved. Stopping the process")
                    break

            self.weights = ideal_weights
            self.bias = ideal_bias
        return loss, self.weights, self.bias

    def predict(self, X):
        # calculating the dot product to predict the results
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        # finding the samples count and the output features count to measure the mean square error
        m, z = X.shape
        q = y.shape
        if len(q) > 1:
            n = q[1]
        else:
            n = 1
        error_generated = (1 / (m * n)) * np.sum((self.predict(X) - y) ** 2)
        return error_generated

    def set_model_params(self, weights, bias):
        # setting the modal parameters to predict the results while evaluating the model
        self.weights = weights
        self.bias = bias
        return