""" 
Like SVMs, Decision Trees are versatile Machine Learning algorithms that can per‐
form both classification and regression tasks, and even multioutput tasks. They are
very powerful algorithms, capable of fitting complex datasets. 


Decision Trees are also the fundamental components of Random Forests,
which are among the most powerful Machine Learning algorithms available
today.

In this section, we will start by 
1 - discussing how to train, visualize, and make predictions with Decision Trees.
2 - we will go through the CART training algorithm used by Scikit-Learn, and we will
discuss how to regularize trees use them for regression tasks
3 - we will discuss some of the limitations of Decision Trees
"""

# Training and visualize a decision tree

""" 
    What need to be decided on:
    - Split feature
    - Split point
    - When to stop splitting
    
    We will first use sklearn then build it from scratch
    The following code trains a DecisionTreeClassifier on the iris dataset
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris_data = load_iris()

X = iris_data.data[:, 2:] # petal length and width (we only take these two columns) # type: ignore
y = iris_data.target # type: ignore


print(iris_data.values) # type: ignore
print(X)
print(y)

""" 
    Construction du modèle
"""

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

""" 
Let's visualize the trained Decision Tree by first using the export_graphviz()
method to output a graph definition file called iris_tree.dot

"""

from sklearn.tree import export_graphviz

# export_graphviz(
# tree_clf,
# out_file="./iris_tree.dot",
# feature_names=iris_data.feature_names[2:], # type: ignore
# class_names=iris_data.target_names, # type: ignore
# rounded=True,
# filled=True
# )

# from subprocess import check_call
# check_call(['dot','-Tpng','iris_tree.dot','-o','iris_tree.png'])

""" 
    The previous method did not finish well!
    
    I will correct it later and try to build it from scratch with a video
"""

""" 
    Training 
    
    Given the whole dataset,
    
    . calculate information gain with each possible split (feature)
    . divide set with that feature and value that gives the most IG (Information Gain)
    . divide tree and do the same for all created branches ...
    . ...until a stopping criteria is reached.
    
    
    Testing
    
    Given a data point: 
    . follow the tree until you reach a leaf node
    . return the most common class label
    
    
    
    Terms 
    
    . Information Gain : IG = E(parent) - [weighted average]. E(children) where E is the Enthropy
    . Entropy E = - sum(p(X). log2(p(X))) and p(X) = #x / n (number of occurences of x divided by the total)
    . Stopping criteria : maximum depth, minimum number of samples, min impurity decrease
    
    
    We are going to create 2 classes: Node and Decision Tree
"""
import numpy as np

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # le * parmi les paramètres signifie que quand je vais vouloir initialiser le node,
        # je dois donner la valeur de value en précisant value ie value devient un parmètre nommé
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = None

    # check if a node is a leaf
    
    def is_a_leaf(self):
        return self.value is not None
    

class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        # stopping citeria
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        # we are going to build the tree recursively using the private method _grow_tree()
    
    def fit(self):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y) # type: ignore
        
        
    def _grow_tree(self, X, y):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y)) # la méthode unique élimine les doublons de la liste des labels qui est y
        
        # check the stopping criteria
        
        
        # find the best split
        
        
        # create child nodes by calling the method again (recursivity)
        
        
        
        
        
    
    def predict(self):
        pass 