import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

"""
    Exercise 1 : Build a model fpr forcasting the daily volume of trash 
    pickups request using linear regression
"""

# Load the dataset
data = pd.read_csv("daily_trash_pickup_dataset.csv")

"""
features: Temperature, Weather, Day_of_Week, Public_Holidays,
Daily_Request_Volume



    1. Data Preparation: 
•	Ensure our dataset is properly formatted and clean 
•	Remove any irrelevant columns and handle missing values appropriately. 
•	Ensure that the label, "Total trash pickup requests for the day," is available and numeric.


"""
# Getting the information of the dataset
print(data.info())

"""

Data columns (total 5 columns):
 #   Column                Non-Null Count    Dtype
---  ------                --------------    -----
 0   Temperature           1000000 non-null  float64
 1   Weather               1000000 non-null  object
 2   Day_of_Week           1000000 non-null  object
 3   Public_Holiday        1000000 non-null  int64
 4   Daily_Request_Volume  1000000 non-null  int64
dtypes: float64(1), int64(2), object(2)
memory usage: 38.1+ MB
None
Temperature             0
Weather                 0
Day_of_Week             0
Public_Holiday          0
Daily_Request_Volume    0
dtype: int64

"""

#  the description of the dataset

print(data.describe())

"""
             Temperature  Public_Holiday  Daily_Request_Volume
count  1000000.000000  1000000.000000        1000000.000000
mean        27.504545        0.100279              0.500486
std          7.499325        0.300372              0.500000
min         15.000000        0.000000              0.000000
25%         21.000000        0.000000              0.000000
50%         27.000000        0.000000              1.000000
75%         34.000000        0.000000              1.000000
max         40.000000        1.000000              1.000000
Temperature             0
Weather                 0
Day_of_Week             0
Public_Holiday          0
Daily_Request_Volume    0
dtype: int64

"""

#  1-  check for missing values in each column

print(data.isnull().sum())

# no missing values 

# 2-  cleaning data