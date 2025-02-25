from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

iris_dataset = load_iris()

X = iris_dataset.data[:, 1]
y = iris_dataset.data[:, 2]

train_split,test_split,train_y_split,test_y_split = train_test_split(X,y,test_size=0.1, random_state=42)

m = test_y_split.shape
test_y_split = test_y_split.reshape(m[0],)

m = test_split.shape
test_split = test_split.reshape(m[0],1)

npzfile = np.load("regression3.npz")

model = LinearRegression(max_epochs=100)

model.set_model_params(npzfile["x1"], npzfile["y1"])

output = pd.DataFrame()
output["Actual values"] = test_y_split
output["predicted values"] = model.predict(test_split).round()
print("Results")
# Error
print("Mean Squared Error", model.score(test_split, test_y_split))
