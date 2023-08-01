"""  
    A key task in machine learning is to take a model and the data to uncover
the values of the model's hidden variables θ given the observed variables x.


Gaussian mixture models work great on clusters with ellipsoidal shapes, but if you try
to fit a dataset with different shapes, you may have bad surprises.
"""




import numpy as np
import pandas as pd

housing = pd.read_csv("../housePrices_train.csv")
print(housing.shape)


class NaiveBayes():

    # we do not need an __init__ method because there is no parameter to initialize

    # the fit method get the training sample X and the training label y

    def fit(self, X, y):
        # get the number of samples and the number of feature
        # with the assumption that X and y are already numpy arrays
        # X.shape returns a tuple with the number or rows and the number of features.
        n_samples, n_features = X.shape

        # get the number of unique classes and store it into classes attribute
        self._classes = np.unique(y)

        # get the number of classes
        n_classes = len(self._classes)

        # First step : Training
        # calculate the value of the mean, variance and prior(frequency) for each class

        # initialize the mean, the variance and the prior
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._variance = np.zeros((n_classes, n_features), dtype=np.float64)
        # for each class we want to have a prior
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for index, a_class in enumerate(self._classes):
            # we only want the sample of the class
            X_a_class = X[y == a_class]
            self._mean[index, :] = X_a_class.mean(axis=0)
            self._variance[index, :] = X_a_class.var(axis=0)
            self._priors[index] = X_a_class.shape[0] / float(n_samples)

    # the predict method get the test Sample

    def predict(self, X):
        y_predict = [self._predict(x) for x in X]
        return np.array(y_predict)

    def _predict(self, x):
        # we want to calculate the posteriors

        posteriors = []
        # calculate posterior probability for each class

        for index, a_class in enumerate(self._classes):
            prior = np.log(self._priors[index])
            posterior = np.sum(np.log(self._pdf(index, x)))  # type: ignore
            posterior += prior
            posteriors.append(posterior)

        # return the class with the hight posterior

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_index, x):
        # let's break the formula of the Gaussian model
        mean = self._mean[class_index]
        variance = self._variance[class_index]

        # first part of the formula
        numerator = np.exp((-(x - mean) ** 2) / (2 * variance))

        # second part
        denominator = np.sqrt(2 * np.pi * variance)

        return numerator/denominator



# now test the code with the test set


from sklearn.model_selection import train_test_split
from sklearn import datasets
    
    
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
    
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=1234)
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    
nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
    
print(f"Naive Bayes classification accuracy {accuracy(y_test, predictions)}")
    