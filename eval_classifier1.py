from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec

data_set = load_iris()

# taking the sepal length and sepal width to classification
X = data_set.data[:, :2]
# converting data binary format
y = (data_set.target != 0) * 1

s = StandardScaler()
x = s.fit_transform(X)

train_split,test_split,train_y_split,test_y_split = train_test_split(x,y,test_size=0.1, random_state=42)

lr = LogisticRegression()

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

lr.fit(train_split, train_y_split)

predicted_values = lr.predict(test_split)

score = lr.accuracy_score_calc(predicted_values, test_y_split)

print("Accuracy score", score)

ax = plt.subplot(gs[0, 0])
fig = plot_decision_regions(X=X, y=y, clf=lr, legend=2)
plt.title("Logistic Regression")
plt.show()

