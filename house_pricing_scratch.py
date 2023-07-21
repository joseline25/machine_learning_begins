""" 
predict price of house base on a dataset with examples having 18 features.

for this task, here is the process to create the madel.

    - get the data : the csv file in a folder 
    
"""

from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

# Import the dataset.

HOUSING_PATH = os.path.join("datasets", "exercise")


csv_path_one = os.path.join(os.getcwd(), HOUSING_PATH)
csv_path = os.path.join(csv_path_one, "housePrices_train.csv")
print(pd.read_csv(csv_path))

housing = pd.read_csv(csv_path)


""" 
Simple Linear Regression : predict the label with one feature
"""

my_feature = "LotArea"
my_label = "SalePrice"

x = np.array( housing[my_feature].values.tolist()).reshape((-1, 1))
y = housing[my_label].values

print(x)

model = LinearRegression()

model = LinearRegression().fit(x, y) # type: ignore

r_sq = model.score(x, y) # type: ignore
print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

y_pred = model.predict(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(f"predicted response:\n{y_pred}")
print(len(y_pred))

y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response:\n{y_pred}")

"""Step:3 Splitting our data"""



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""Step:4 Training the model"""

regressor = LinearRegression() 

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(f"second predicted response:\n{y_pred}")




"""Step:6 Plotting regression line"""

plt.scatter(X_train, y_train,color='r') 

plt.plot(X_test, y_pred,color='k') 

plt.show()


"""  
    Multiple Linear Regression With scikit-learn
You can implement multiple linear regression following the same steps 
as you would for simple regression. The main difference is that your x 
array will now have two or more columns.

"""
feature_one = "LotArea"
feature_two = "GarageArea"
feature_three = "MoSold"
x = np.array( housing[[feature_one,feature_two, feature_three]].values.tolist())
y = housing[my_label].values

print(x)

model = LinearRegression().fit(x, y) # type: ignore

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")

print(f"coefficients: {model.coef_}")

"""Step:4 Training the model"""

regressor = LinearRegression() 

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(f"second predicted response:\n{y_pred}")


"""Step:6 Plotting regression line"""

plt.scatter(X_train, y_train,color='r') 

plt.plot(X_test, y_pred,color='k') 

plt.show()
