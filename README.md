# LinearRegression 

In this project we are using the iris flower data. Here we are trying to use different combinations of the data with the linear regression to see which is giving the best results.

We have loaded the data from the datasets library available in the python.

This iris dataset contains four features sepal length, sepal width, petal length and petal width. It has a data array and a target array. Target array consists of the flower type for the given feature values.

Correlation matrix of the dataset:

<img width="468" alt="image" src="https://github.com/user-attachments/assets/ade9385e-9154-487b-bbde-a06cc04c52ee" />
 

The correlation matrix gives the correlation between all the attributes. The correlation coefficient value will be between the -1 and 1. -1 indicates the negative correlation and +1 indicates the positive correlation. If the value is zero, then there is no relation between the features.

From the above matrix we can say that the features 3 and 4 are highly positively correlated. i.e. petal length and petal width have high dependency with each other.



Models:
I have implemented all the training models in separate files named as linear_regression(model_number).py and stored the parameter values in the .npz files. I have created the evaluation files for testing all the trained models and loaded the saved parameter data from the .npz files and tested them.


All features vs the target variable (flower name)
	
In this model I have used all four features as independent variable to predict the type of flower by using the implemented linear regression methods.

Mean square error:

0.07207915186854978

Graph for the loss

 
Models with Different feature combinations:

1. petal length (cm) petal width (cm) vs sepal width (cm)"

In this model I have used the petal length and width to predict the sepal width for the flowers.

Mean Square Error:

Score - 1.9346131326234475

Loss Graph:

 

2.  Sepal width vs petal width

In this model I have used the sepal width to predict the petal width.

Mean Squared error:

Score - 0.7244571941725192

Loss Graph:

 
3. Sepal width vs petal length

Mean squared Error:

Score - 4.327956941437939

Loss Graph:

 

4. Sepal length, Sepal width vs Petal length

Mean Squared Error:

Score - 1.5456121515438492

Loss Graph:

 

Regularization:

I have used the third model from the trained models and added the regularization of 0.8 and recorded the values again.

Mean Square Errors:

Before score - 4.327956941437939

After score  - 2.9572118170509554

Loss:

Before vs After graph
  

After using a 0.5 regularization on the previous model the mean square error values is decreased from 4.32 to 2.95.

This is because adding the regularization value makes the penalty for high valued weights more significant and the modal tries to learn more simpler pattern achieving less error on the validation data.

Regularization also avoids overfitting to the trained data and increases the performance of the modal on the validation data.

Regression with Multiple Outputs
In this model I have used the both sepal length and the sepal width to predict the both petal length and the petal width.

Error function used 
Error = 1/(mn) * np.sum((y_pred – y_actual) ** 2)
Mean squared error.
0.9417018206751778
Loss Graph:
 

Logistic Regression:

Implemented the 3 models with different set of features from the iris dataset.

1.Classification based on the sepal length and sepal width

Accuracy score:  0.4 



2.Classification based on petal length and petal width

Accuracy score: 0.4

 
3.Using all the features

Accuracy: 0.4

