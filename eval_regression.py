import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

npzfile = np.load("regression.npz")

model = LinearRegression(max_epochs=100)
model.set_model_params(npzfile["x1"], npzfile["y1"])

data_set = load_iris()

X = data_set.data
y = data_set.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

output = pd.DataFrame()
output["actual data"] = y_test
output["predictions"] = model.predict(X_test).round()
print("Result -")
print(output)
# error
print("Mean Squared Error", model.score(X_test, y_test))
