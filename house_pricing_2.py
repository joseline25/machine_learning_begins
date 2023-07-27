import pandas as pd
import matplotlib.pyplot as plt


# get the data from the csv file
housing = pd.read_csv('housePrices_train.csv')
print(housing)

# visualize the data with the features OverallQual as x and SalePrice as y
plt.scatter(housing.OverallQual, housing.SalePrice)
plt.show()

# visualize the data with the features GrLivArea as x and SalePrice as y 
# ( looks more like a line)
plt.scatter(housing.GrLivArea, housing.SalePrice)
plt.show()

# visualize the data with the features GarageCars as x and SalePrice as y
plt.scatter(housing.GarageCars, housing.SalePrice)
plt.show()

# choose my feature and my label

my_feature = "OverallQual"
my_label = "SalePrice"

# Construction of the loss function

# y = m.x + b

def loss_function(m, b, points):
    
    # initialize the total_errors
    total_errors = 0
    
    for i in range(len(points)):
        x = points.iloc[i][my_feature]
        y = points.iloc[i][my_label]
        
        total_errors += (y - (m * x + b)) ** 2
    total_errors /= float(len(points))
    
    
#implement the gradient descend

def gradient_descend(m_now, b_now, points, L):
    
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    for i in range(n):
        x = points.iloc[i][my_feature]
        y = points.iloc[i][my_label]
        
        m_gradient += (-2/n) * x * (y - (m_now * x + b_now))
        b_gradient += (-2/n ) * (y - (m_now * x + b_now))
        
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    
    return m, b



# initialize the parameters

m = 0
b = 0
# the learning rate
L = 0.001

# the number of iterations 
epochs = 300

for i in range(epochs):
    if i % 10 == 0:
        print(f"epochs : {i}")
    m,b = gradient_descend(m, b, housing, L)
    
print(f' m : {m} and b : {b}')

plt.scatter(housing[my_feature], housing[my_label], color="purple")
plt.plot(list(range(0, 20)), [m*x+b for x in range(0,20)], color="red")
plt.show()
    
    