from sklearn.metrics import mean_squared_error, mean_absolute_error
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

housing = pd.read_csv("../housePrices_train.csv")

print(housing)

# implement multilinear regression from scratch
# predict from multiple features

# 1- get the correlation between each integer values and the label we want to predict

my_label = "SalePrice"
correlations = housing.corr(numeric_only=True)[my_label]
print("the different correlations are: ")
print(correlations)
# after printing correlations, we will choose as features
# feature_one = "TotalBsmtSF"
# feature_two = "1stFlrSF"
# feature_three = "GrLivArea"
# feature_four = "OverallQual"

# represent our features with our label


saleprices = housing["SalePrice"]
# choose some columns in the dataset
housing_second = housing[["TotalBsmtSF", "1stFlrSF",
                          "GrLivArea", "OverallQual", "YearBuilt"]]
# add saleprices to housing_second
housing_second["saleprices"] = saleprices
print(housing_second)

# we are going to work with housing_second

# let plot first what we have

plt.scatter(housing_second["TotalBsmtSF"], housing_second["saleprices"])
# name the axis
plt.xlabel('Total of BsmtSF')
plt.ylabel('Sale Prices')
plt.show()

# transform housing_second into a numpy array

housing_second_np = housing_second.to_numpy()
housing_second_np.shape


X_train, y_train = housing_second_np[:, :5], housing_second_np[:, 5]
print(X_train.shape)
print(y_train.shape)
print(X_train)


model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_train)
print(predictions)

# plt.scatter(X_train, predictions)
# plt.plot(X_train, y_train, color="r")
# plt.show()

# mean squared errors functions


mse = mean_squared_error(predictions, y_train)
mae = mean_absolute_error(predictions, y_train)

print(mse, mae)

predictions_housing = pd.DataFrame({'TotalBsmtSF': housing_second["TotalBsmtSF"],
                                    '1stFlrSF': housing_second["1stFlrSF"],
                                    'GrLivArea': housing_second["GrLivArea"],
                                    'OverallQual': housing_second["OverallQual"],
                                    'YearBuilt': housing_second["YearBuilt"],
                                    'Predictions': predictions})

print(predictions_housing)

def get_predictions(model, X):
    pass