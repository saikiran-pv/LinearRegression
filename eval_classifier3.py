from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

data_set = load_iris()

X = data_set.data

# converting data binary format
y = (data_set.target != 0) *  1

s = StandardScaler()
x = s.fit_transform(X)

train_split,test_split,train_y_split,test_y_split = train_test_split(x,y,test_size=0.1, random_state=42)

lr = LogisticRegression()

lr.fit(train_split, train_y_split)

predicted_values = lr.predict(test_split)

score = lr.accuracy_score_calc(predicted_values, test_y_split)

print("Accuracy score", score)