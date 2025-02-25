from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

iris_data_set = load_iris()

X = iris_data_set.data
y = iris_data_set.target

# finding the correlation between the features
correlation_matrix_between_features = np.corrcoef(X, rowvar=False)
print("Correlation coefficient matrix:")
print(correlation_matrix_between_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression(max_epochs=100)
loss, weights, bias = model.fit(X_train, y_train)

np.savez("regression.npz", x1 = weights, y1 = np.array([bias]))

plt.title("Loss, all features vs target variable")
plt.plot(loss)
plt.show()