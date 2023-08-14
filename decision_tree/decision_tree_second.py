"""  
    Decision Trees
A Decision Tree is a tree-like model that predicts the value of a target 
variable based on several input variables. It splits the data based on the
values of the input variables, creating a tree-like structure. 
The leaves of the tree contain the predicted values.

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

# these two next lines are necessary for the library graphviz to run
# on top of installing it in the virtual env of the project, we also had
# to install it on the whole computer 
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

data = load_iris()
X = data.data # type: ignore
y = data.target # type: ignore

print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit the data to the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# We can now use the trained classifier to predict the class of the test data
y_pred = clf.predict(X_test)

# Finally, we can evaluate the performance of the classifier using accuracy, 
# precision and recall

from sklearn.metrics import accuracy_score, precision_recall_curve

print("Accuracy:", accuracy_score(y_test, y_pred))


# precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# print("Precision:", precision)
# print("Recall:", recall)
# print("Thresholds:", thresholds)

# Visualize the tree using graphviz

from sklearn.tree import export_graphviz
import graphviz


dot_data = export_graphviz(clf, out_file=None, feature_names=data.feature_names, class_names=data.target_names) # type: ignore
graph = graphviz.Source(dot_data)
graph.render("./iris")

# This will create a visualization of the Decision Tree in the file “iris.pdf”.


