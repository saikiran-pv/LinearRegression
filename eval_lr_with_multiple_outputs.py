from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

npzfile = np.load("lr_with_multiple_outputs.npz")

model = LinearRegression(max_epochs=100)
model.set_model_params(npzfile["x1"], npzfile["y1"])

iris_dataset = load_iris()

X = iris_dataset.data[:,:2]
y = iris_dataset.data[:,2:4]

train_split,test_split,train_y_split,test_y_split = train_test_split(X,y,test_size=0.1, random_state=42)

print(model.predict(test_split).round())
print("Mean Squared Error :-", model.score(test_split, test_y_split))