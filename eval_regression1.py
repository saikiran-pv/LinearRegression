import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

npzfile = np.load("regression1.npz")

model2 = LinearRegression(max_epochs=100)

model2.set_model_params(npzfile["x1"], npzfile["y1"])

iris_data_set = load_iris()

X = iris_data_set.data[:,2:4]
y = iris_data_set.data[:, 1]

train_split,test_split,train_y_split,test_y_split = train_test_split(X,y,test_size=0.1, random_state=42)

output = pd.DataFrame()
output["actual"] = test_y_split
output["predictions"] = model2.predict(test_split).round()
print("Results")
print(output)
# Error
print("Mean Squared Error :- ", model2.score(test_split, test_y_split))