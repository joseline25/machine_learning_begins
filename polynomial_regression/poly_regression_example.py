import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)


#plot X, y
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.show()

#implement polynomial regression of X,y

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=3, include_bias=False) 
# par défaut, le degré est égal à 2
X_poly = poly_features.fit_transform(X) 

# si degree = 2, X_poly[0] est un vecteur à deux élements avec X[0] coe premier élément
# et le second élément?

# X_poly now contains the original feature of X plus the square of this feature.
# Now you can fit a LinearRegression model to this extended training data
 
 
# si degree = 3, ce vecteur contient les 3 valeurs de X[0], X[0]**2  et X[0]**3
#[1.36911617 1.8744791  2.56637965]

# A partir de ces valeurs, on peut utiliser la classe LinearRegression de sklearn
# pour faire de la regression polynomiale

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

# plot X_poly


Y_poly = lin_reg.predict(X_poly)


plt.scatter(X, y)
plt.plot(X, Y_poly, color="r")
plt.show()