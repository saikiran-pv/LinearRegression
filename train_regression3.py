from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

iris_dataset = load_iris()
# sepal width (cm) vs petal length (cm)
X = iris_dataset.data[:, 1]
y = iris_dataset.data[:, 2]

train_split, test_split, train_y_split, test_y_split = train_test_split(X,y, test_size =0.1, random_state=42)

m = train_split.shape
train_split = train_split.reshape(m[0],1)

model = LinearRegression(max_epochs=100)
loss, weights, bias = model.fit(train_split, train_y_split)

np.savez("regression3.npz", x1 = weights, y1 = np.array([bias]))

print(loss)
plt.plot(loss)
plt.title("Loss, sepal width (cm) vs petal length (cm)")
plt.xlabel("steps")
plt.ylabel("loss")
plt.show()
