from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

iris_dataset = load_iris()

X = iris_dataset.data[:,:2]
y = iris_dataset.data[:,2]

train_split,test_split,train_y_split,test_y_split = train_test_split(X,y,test_size=0.1, random_state=42)

npzfile = np.load("regression4.npz")

model = LinearRegression(max_epochs=100)

model.set_model_params(npzfile["x1"], npzfile["y1"])

output = pd.DataFrame()
output["actual values"] = test_y_split
output["predicted values"] = model.predict(test_split).round()
print("Results")
# Error
print("Mean Square Error", model.score(test_split, test_y_split))
