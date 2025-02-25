from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
import numpy as np

iris_data_set = load_iris()

X = iris_data_set.data[:,2:4]
y = iris_data_set.data[:, 1]

train_split, test_split, train_y_split, test_y_split = train_test_split(X,y, test_size =0.1, random_state=42)

model = LinearRegression(max_epochs=100)
loss, weights, bias = model.fit(train_split, train_y_split)

np.savez("regression1.npz", x1 = weights, y1 = np.array([bias]))

plt.title("Loss, petal length (cm) petal width (cm) vs sepal width (cm)")
plt.plot(loss)
plt.xlabel("steps")
plt.ylabel("loss")
plt.show()