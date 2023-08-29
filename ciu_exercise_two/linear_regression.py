import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import holidays

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder


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
# a - ensure that the dataset is properly formatted and clean

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

# no missing values in the dataset

# # for columns of type int64 or float64, check if the values can fit int32 using numpy

# Temperature
Temperature_min = np.finfo(np.float32).min
Temperature_max = np.finfo(np.float32).max
is_compatible_Temperature = (data['Temperature'].min() >= Temperature_min) and (
    data['Temperature'].max() <= Temperature_max)
print(is_compatible_Temperature)  # True

data['Temperature'] = data['Temperature'].astype('float32')

# Public_Holiday
Public_Holiday_min = np.iinfo(np.int32).min
Public_Holiday_max = np.iinfo(np.int32).max
is_compatible_Public_Holiday = (data['Public_Holiday'].min() >= Public_Holiday_min) and (
    data['Public_Holiday'].max() <= Public_Holiday_max)
print(is_compatible_Public_Holiday)  # True

data['Public_Holiday'] = data['Public_Holiday'].astype('int32')

# Daily_Request_Volume
Daily_Request_Volume_min = np.iinfo(np.int32).min
Daily_Request_Volume_max = np.iinfo(np.int32).max
is_compatible_Daily_Request_Volume = (data['Daily_Request_Volume'].min() >= Daily_Request_Volume_min) and (
    data['Daily_Request_Volume'].max() <= Daily_Request_Volume_max)
print(is_compatible_Daily_Request_Volume)  # True

data['Daily_Request_Volume'] = data['Daily_Request_Volume'].astype('int32')

print(data.info())  # now every numeric value is on 32 bits

# Clean the data


# get correlation of features with seaborn
# pivot table to compute correlation
pivot_table = data.pivot_table(
    index='Day_of_Week', values='Daily_Request_Volume')

# create the heatmap
sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
plt.show()
# Figure_11


# Manage the Weather feature according to Temperature
# check for mismathch between Temperature and Weather
# create a box plot
sns.boxplot(x='Weather', y='Temperature', data=data)
# set plot labels
plt.xlabel("Weather")
plt.ylabel("Temperature")

plt.show()
# we can observe from the figure  that in average, blow 27 of temperature it is rainy.


# now we will correct the wron entries ( check if ther is a better way of doing this)
mask = (data['Temperature'] < 27)
data.loc[mask, 'Weather'] = data.loc[mask, 'Weather'].replace('Sunny', 'Rainy')

mask_one = (data['Temperature'] >= 27)
data.loc[mask_one, 'Weather'] = data.loc[mask_one,
                                         'Weather'].replace('Rainy', 'Sunny')

sns.boxplot(x='Weather', y='Temperature', data=data)

plt.xlabel("Weather")
plt.ylabel("Temperature")

plt.show()

# show the dataset
print(data)

# Aggregate the data by Day_of_Week and calculate the total daily request volume

data_agg = data.groupby("Day_of_Week").agg(
    {"Weather": "first", "Temperature": "first", "Daily_Request_Volume": "sum"}).reset_index()


print(data_agg) # Good!!! 

# Aggregate the data by Temperature and calculate the total daily request volume

data_agg_two = data.groupby("Temperature").agg(
    {"Daily_Request_Volume": "sum"}).reset_index()

print(data_agg_two)



# print(data_agg_two)

#on va maintenant travailler avec les données aggrégé autour de Day_of_Week où on peut
# drop les features Weather et Temperature

# II - Data visualization or EDA Exploratory Data Analysis

# ce qui est dans le doc d'abord

# visualizing the distribution of trash pickup requests across different days of the week.
# We use a bar plot to show the count of requests for each day

# on the old dataset
plt.figure(figsize=(8,6))
sns.countplot(x='Day_of_Week', data=data, hue='Daily_Request_Volume')
plt.title("Daily Request Volume by Day of Week")
plt.xlabel("Day_of_Week")
plt.ylabel('Daily_Request_Volume_Count')
plt.show()

# on the new dataset
plt.figure(figsize=(8,6))
sns.countplot(x='Day_of_Week', data=data_agg, hue='Daily_Request_Volume')
plt.title("Daily Request Volume by Day of Week")
plt.xlabel("Day_of_Week")
plt.ylabel('Daily_Request_Volume_Count')
plt.show()

# visualize the distribution of trash pickup requests across different weather conditions.
# We use a bar plot again to show the count of requests for each weather condition
# for data without aggregation

plt.figure(figsize=(8,6))
sns.countplot(x='Weather', data=data, hue='Daily_Request_Volume')
plt.title("Daily Request Volume by Weather")
plt.xlabel("Day_of_Week")
plt.ylabel('Daily_Request_Volume')
plt.show()
# We can observe that in general, there are more Daily request when the weather is sunny.
# Also, When it is Sunny, we almost have the same amount of pickup and no pickup
# or it just means that there are more entries in our dataset for sunny weather


# visualize the distribution of trash pickup requests across different temperature ranges.
# We use a histogram to show the frequency of requests for different temperature ranges

# with data
plt.figure(figsize=(8,6))
sns.histplot(x='Temperature', data=data, hue='Daily_Request_Volume', bins=10)
plt.title("Daily Request Volume by Temperature")
plt.xlabel("Temperature")
plt.ylabel('Daily_Request_Volume')
plt.show()

# with data_agg
plt.figure(figsize=(8,6))
sns.histplot(x='Temperature', data=data_agg, hue='Daily_Request_Volume', bins=10)
plt.title("Daily Request Volume by Temperature")
plt.xlabel("Temperature")
plt.ylabel('Daily_Request_Volume')
plt.show()

# with data_agg_two
plt.figure(figsize=(8,6))
sns.histplot(x='Temperature', data=data_agg_two, hue='Daily_Request_Volume', bins=10)
plt.title("Daily Request Volume by Temperature")
plt.xlabel("Temperature")
plt.ylabel('Daily_Request_Volume')
plt.show()


# # Compute the correlation matrix
# corr_matrix = data.corr()

# # Display the correlation matrix using a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()

# Model Building 

# Step 1

# Perform one-hot encoding on categorical features
data_encoded = pd.get_dummies(data, columns=['Day_of_Week', 'Weather'])

print("data encoded ")
print(data_encoded)

# Map True/False values to 1/0 for correlation calculation
data_encoded['Day_of_Week_Friday'] = data_encoded['Day_of_Week_Friday'].astype(int)
data_encoded['Day_of_Week_Monday'] = data_encoded['Day_of_Week_Monday'].astype(int)
data_encoded['Day_of_Week_Tuesday'] = data_encoded['Day_of_Week_Tuesday'].astype(int)
data_encoded['Day_of_Week_Wednesday'] = data_encoded['Day_of_Week_Wednesday'].astype(int)
data_encoded['Day_of_Week_Thursday'] = data_encoded['Day_of_Week_Thursday'].astype(int)
data_encoded['Day_of_Week_Saturday'] = data_encoded['Day_of_Week_Saturday'].astype(int)
data_encoded['Day_of_Week_Sunday'] = data_encoded['Day_of_Week_Sunday'].astype(int)
print(data_encoded)

# nomalize these data



train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
print(train_set)

my_feature = "Temperature"
my_label = "Daily_Request_Volume"


x = np.array( data[my_feature].values.tolist()).reshape((-1, 1))
y = data[my_label].values

model = LinearRegression()

model = LinearRegression().fit(x, y) 

r_sq = model.score(x, y)

print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

y_pred = model.predict(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(f"predicted response:\n{y_pred}")
print(len(y_pred))

y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response:\n{y_pred}")






# Select only the numerical features
numerical_features = ['Temperature', 'Public_Holiday', 'Daily_Request_Volume']

# Subset the dataset to include only numerical columns
numerical_data = data[numerical_features]

# Compute the correlation matrix
corr_matrix = numerical_data.corr()

# Highlighted column
highlight_col = 'Daily_Request_Volume'

# Create a mask to highlight the specified column
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
mask[:, numerical_features.index(highlight_col)] = False

# Display the correlation matrix using a heatmap with the highlighted column
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
            vmin=-1, vmax=1, center=0, cbar_kws={"shrink": 0.7}, mask=mask)
plt.title('Correlation Matrix (Highlighted Column: {})'.format(highlight_col))
plt.show()

#c'est pas top

