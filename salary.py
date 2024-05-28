#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset

Dataset = pd.read_csv('Salary_Data.csv')
x = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, 1].values
#print(x)
#print(y)

#splitting the dataset into training and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#training simple linear regression model on the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

#predicting the test set results

y_pred = regressor.predict(x_test)

#print(y_pred)

# Visualising the Training set results

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


