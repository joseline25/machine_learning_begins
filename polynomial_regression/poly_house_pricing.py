import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

# NeuralNine tutorial

# get the data

housing = pd.read_csv("../housePrices_train.csv")
# table des coefs de correlation
my_feature = "YearBuilt"
my_label = "SalePrice"

for i in range(len(housing.columns.values)):
    if housing.dtypes.values[i] == 'int64':
        print(
            f'Feature : {housing.columns.values[i]}   Correlation coeficient : {scipy.stats.pearsonr(housing[housing.columns.values[i]].values.tolist(), housing[my_label])[0]}')


X = np.array(housing[my_feature].values.tolist()).reshape(-1, 1)
print(X)

Y = np.array(housing[my_label].values.tolist()).reshape(-1, 1)

#plotting the points 

# plt.scatter(X, Y) # type: ignore
# plt.show()

# I- création du modèle de regression linéaire
# 1- instanciation du modèle 

reg = LinearRegression()
reg.fit(X, Y)

# plot the linear regression model
# plot la droit de regression avec la methode de prediction

X_vals = X
Y_vals = reg.predict(X_vals)
print(reg.intercept_)
print(reg.coef_)


plt.scatter(X, Y)
plt.plot(X_vals, Y_vals, color="r")
plt.show()

# II- Polynomial regression
# il n'y a pas une classe de sklearn pour la regression linéaire.
# ce qu'on fait c'est qu'on utilise quelque chose qu'on appelle polynomial features
from sklearn.preprocessing import PolynomialFeatures

# do not take a degree too high or we will have the overfitting problem

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# plot the polynomial model

reg = LinearRegression()
reg.fit(X_poly, Y)
X_vals_poly = poly_features.transform(X_vals)
Y_vals_poly = reg.predict(X_vals_poly)

plt.scatter(X, Y)
plt.plot(X_vals, Y_vals_poly, color="r")
plt.show()
print(reg.intercept_)
print(reg.coef_)

