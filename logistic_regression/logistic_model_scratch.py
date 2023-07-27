import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt




def sigmoid(x):
    return 1/(1+ np.exp(-x))
    

class LogisticRegression():
    def __init__(self, learning_rate=0.001, number_iters=1000):
        self.learning_rate = learning_rate
        self.number_iters = number_iters
        self.weight = None
        self.bias = None
        
        # 2) Given a data point , 
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # initialize weight and bias 
        
        self.weight = np.zeros(n_features) 
        self.bias = 0

        # repeat n times
        for _ in range(self.number_iters):
            #predict result 
            linear_pred = np.dot(X, self.weight) + self.bias
            predictions = sigmoid(linear_pred)
            
            # calculate the gradient to figure out new weight and bias values (J'(O))
            dw = 1/n_samples * np.dot(X.T, (predictions - y)) # X.T est la transposée de X
            db = 1/n_samples * np.sum(predictions - y)
            
            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
        
     # 3- testing
    def predict(self, X):
        # put the values from the data point into the equation 
        linear_pred = np.dot(X, self.weight) + self.bias # type: ignore
        y_pred = sigmoid(linear_pred)
        
        # choose the label based on the probability
        
        class_prediction = [0  if y< 0.5 else 1 for y in y_pred]
        return class_prediction
        

# train the model

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target  # type: ignore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf = LogisticRegression(learning_rate=0.01)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)