""" 
    Random Forests
Random Forests are a powerful machine learning algorithm that uses multiple Decision 
Trees to make predictions. Each Decision Tree is trained on a random subset of the 
data and a random subset of the input variables. The final prediction is made by 
taking the average of the predictions of all the Decision Trees.

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data # type: ignore
y = data.target # type: ignore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)


# We can now use the trained classifier to predict the class of the test data

y_pred = clf.predict(X_test)

# we can evaluate the performance of the classifier using accuracy

print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualization unsing graphviz

from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(clf.estimators_[0], out_file=None, feature_names=data.feature_names, class_names=data.target_names) # type: ignore
graph = graphviz.Source(dot_data)
graph.render("tree")

#This will create a visualization of the first Decision Tree in the Random Forest
# in the file “tree.pdf”