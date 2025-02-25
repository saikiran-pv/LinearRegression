from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

iris_dataset = load_iris()

X = iris_dataset.data[:,:2]
y = iris_dataset.data[:,2:4]

train_split,test_split,train_y_split,test_y_split = train_test_split(X,y,test_size=0.1, random_state=42)
m = train_y_split.shape
train_y_split.reshape(m[0],2)

m = train_split.shape
train_split.reshape(m[0],2)
n = train_y_split.shape
train_y_split.reshape(n[0],2)
model = LinearRegression(max_epochs=100)

loss, weights, bias = model.fit(train_split, train_y_split)

np.savez("lr_with_multiple_outputs.npz", x1 = weights, y1 = np.array([bias]))

plt.plot(loss)
plt.title("Loss, sepal length sepal width vs petal length petal width")
plt.show()