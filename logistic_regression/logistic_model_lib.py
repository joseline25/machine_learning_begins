from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
print(list(iris.keys()) )# type: ignore
print(list(iris.values())) # type: ignore

X = iris["data"][:, 3:] # type: ignore # petal width
y = (iris["target"] == 2).astype(np.int16)  # type: ignore # 1 if Iris-Virginica, else 0

# Now let’s train a Logistic Regression model

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

# NumPy’s reshape() function allows one dimension to be –1, which means “unspecified”: the value is inferred
# from the length of the array and the remaining dimensions.

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.show()