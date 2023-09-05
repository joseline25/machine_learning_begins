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
data = pd.read_csv("individual_trash_pickup_dataset.csv")

print(data)

# plot the data dataset
data.plot(kind='scatter', x='Day_of_Week',
                          y='Resident_Trash_Pickup_Request')

plt.xlabel('Day_of_Week')
plt.ylabel('Resident_Trash_Pickup_Request')
plt.show()
"""
features: Date,Resident_ID,Temperature,Weather,Day_of_Week,Previous_Requests,
Public_Holiday,Resident_Trash_Pickup_Request


    1. Data Preparation: 
•	Ensure our dataset is properly formatted and clean 
•	Remove any irrelevant columns and handle missing values appropriately. 
•	Ensure that the label, "Total trash pickup requests for the day," is available and numeric.


"""

# Getting the information of the dataset

print(data.info())

# the types are object(3) for Date, Weather, Day_of_Week, Previous_Requests
# and int64 for the rest(5). The size of the dataset is 1.000.000
"""
    RangeIndex: 1000000 entries, 0 to 999999
Data columns (total 8 columns):
 #   Column                         Non-Null Count    Dtype
---  ------                         --------------    -----
 0   Date                           1000000 non-null  object
 1   Resident_ID                    1000000 non-null  int64
 2   Temperature                    1000000 non-null  int64
 3   Weather                        1000000 non-null  object
 4   Day_of_Week                    1000000 non-null  object
 5   Previous_Requests              1000000 non-null  int64
 6   Public_Holiday                 1000000 non-null  int64
 7   Resident_Trash_Pickup_Request  1000000 non-null  int64
dtypes: int64(5), object(3)
memory usage: 61.0+ MB
None
Date                             0
Resident_ID                      0
Temperature                      0
Weather                          0
Day_of_Week                      0
Previous_Requests                0
Public_Holiday                   0
Resident_Trash_Pickup_Request    0
dtype: int64

"""

#  the description of the dataset

print(data.describe())

"""
     Resident_ID     Temperature  Previous_Requests  Public_Holiday  Resident_Trash_Pickup_Request
count  1000000.000000  1000000.000000     1000000.000000  1000000.000000                 1000000.000000
mean        48.492014       27.504545           5.005860        0.100279                       0.500486
std         29.459271        7.499325           3.160921        0.300372                       0.500000
min         -1.000000       15.000000           0.000000        0.000000                       0.000000
25%         23.000000       21.000000           2.000000        0.000000                       0.000000
50%         48.000000       27.000000           5.000000        0.000000                       1.000000
75%         74.000000       34.000000           8.000000        0.000000                       1.000000
max         99.000000       40.000000          10.000000        1.000000                       1.000000
Date                             0
Resident_ID                      0
Temperature                      0
Weather                          0
Day_of_Week                      0
Previous_Requests                0
Public_Holiday                   0
Resident_Trash_Pickup_Request    0
dtype: int64

"""


#    check for missing values in each column

print(data.isnull().sum())

# no missing values

# for columns of type int64, check if the values can fit int32 using numpy
# Resident_ID
Resident_ID_min = np.iinfo(np.int32).min
Resident_ID_max = np.iinfo(np.int32).max
is_compatible_Resident_ID = (data['Resident_ID'].min() >= Resident_ID_min) and (
    data['Resident_ID'].max() <= Resident_ID_max)
print(is_compatible_Resident_ID)  # True
# Temperature
Temperature_min = np.iinfo(np.int32).min
Temperature_max = np.iinfo(np.int32).max
is_compatible_Temperature = (data['Temperature'].min() >= Temperature_min) and (
    data['Temperature'].max() <= Temperature_max)
print(is_compatible_Temperature)  # True

# we find that it is also True for Previous_Requests, Public_Holiday, Resident_Trash_Pickup_Request

# let's convert
data['Resident_ID'] = data['Resident_ID'].astype('int32')
data['Temperature'] = data['Temperature'].astype('int32')
data['Previous_Requests'] = data['Previous_Requests'].astype('int32')
data['Public_Holiday'] = data['Public_Holiday'].astype('int32')
data['Resident_Trash_Pickup_Request'] = data['Resident_Trash_Pickup_Request'].astype(
    'int32')

print(data.info())  # ça a marché!!!

# get the correlation between Day_of_Week and Resident_Trash_Pickup_Request
correlation = data['Day_of_Week'].astype('category').cat.codes.corr(
    data['Resident_Trash_Pickup_Request'])

print(correlation)  # -0.0005699850232853372

# with seaborn
# pivot table to compute correlation
pivot_table = data.pivot_table(
    index='Day_of_Week', values='Resident_Trash_Pickup_Request')

# create the heatmap
sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
plt.show()


"""
to convert True/False values to 1/0, 
data['Resident_Trash_Pickup_Request'] = data['Resident_Trash_Pickup_Request'].astype(int)
"""

# correlation between all the features
# convert the date for the correlation_matrix
data['Date'] = pd.to_datetime(data['Date'])
# convert categorical data to numerial by one-hot encoding avec pd.get_dummies()
# or hot encode with OneHotEncoder de sklearn
data_encoded = pd.get_dummies(data, columns=['Weather', 'Day_of_Week'])

correlation_matrix = data_encoded.corr()

# increase the size of the picture
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# 2-  cleaning data

# check for mismathch between Temperature and Weather
# create a box plot
sns.boxplot(x='Weather', y='Temperature', data=data)
# set plot labels
plt.xlabel("Weather")
plt.ylabel("Temperature")

plt.show()
# we can observe from the figure  that in average, blow 27 of temperature it is rainy.

# now we will correct the wrong entries ( check if ther is a better way of doing this)
mask = (data['Temperature'] < 27)
data.loc[mask, 'Weather'] = data.loc[mask, 'Weather'].replace('Sunny', 'Rainy')

mask_one = (data['Temperature'] >= 27)
data.loc[mask_one, 'Weather'] = data.loc[mask_one,
                                         'Weather'].replace('Rainy', 'Sunny')

sns.boxplot(x='Weather', y='Temperature', data=data)

plt.xlabel("Weather")
plt.ylabel("Temperature")

plt.show()

# correct the mismatch between the date feature and the public_holiday feature
# according to the cameroonian calendar
cameroon_holidays = holidays.country_holidays('CM')
list_cm_holidays = []
for i in data['Date']:
    if i in cameroon_holidays:
        list_cm_holidays.append(1)
    else:
        list_cm_holidays.append(0)
data['Public_Holiday'] = list_cm_holidays

# this process works but takes some times since we have 1.000.000 entries


# Aggregate the data by date and calculate the total daily request volume
aggregated_data = data.groupby("Date").agg({"Resident_ID": "first", "Weather": "first",
                                            "Day_of_Week": "first", "Previous_Requests": "sum",
                                            "Public_Holiday": "first", "Resident_Trash_Pickup_Request": "sum"}).reset_index()

# Print the aggregated data
# print(aggregated_data)

aggregated_data_two = data.groupby(
    'Date')['Resident_Trash_Pickup_Request'].sum().reset_index()
print(aggregated_data_two)

# get the unique date
uniques_dates = data['Date'].unique()
print(uniques_dates)
data['Date'] = pd.to_datetime(data['Date']).dt.date

aggregated_data_two = data.groupby(
    'Date')['Resident_Trash_Pickup_Request'].sum().reset_index()
print(aggregated_data_two)  # good!!!!

# group by date when resident_trash_pickup_request == 1
aggregated_data_good = data[data['Resident_Trash_Pickup_Request'] == 1].groupby(
    'Date')['Resident_Trash_Pickup_Request'].sum().reset_index()
print(aggregated_data_good)
# it gives the same as the prvious one and it is exactly 11 requests for the first date

aggregated_data_good = data.groupby('Date').agg(Resident_Trash_Pickup_Request_Sum=(
    'Resident_Trash_Pickup_Request', 'sum'), Date=('Date', 'first'), Public_Holiday=('Public_Holiday', 'first'), Day_of_Week=('Day_of_Week', 'first'))

# we can not put resident_id, weather et previous request
print(aggregated_data_good)

# plot the aggregated_data_good dataset
aggregated_data_good.plot(kind='scatter', x='Day_of_Week',
                          y='Resident_Trash_Pickup_Request_Sum')

plt.xlabel('Day_of_Week')
plt.ylabel('Resident_Trash_Pickup_Request_Sum')
plt.show()

# Extract features and target variable from aggregated_data_good
numeric_features = ["Public_Holiday"]
non_numeric_features = ["Date", "Day_of_Week"]
target = aggregated_data_good["Resident_Trash_Pickup_Request_Sum"]

# Perform feature scaling
# separate the dataset between numeric features and non numeric features
aggregated_data_good_numeric = aggregated_data_good[numeric_features]
aggregated_data_good_non_numeric = aggregated_data_good[non_numeric_features]

scaler = StandardScaler()
# scale for numerical features

aggregated_data_good_scaled = pd.DataFrame(scaler.fit_transform(
    aggregated_data_good_numeric), columns=numeric_features)
# aggregated_data_good.transform(
# aggregated_data_good_numeric, columns=numeric_features)


# combine scaled numeric features and non-numeric features
data_preprocessed = pd.concat(
    [aggregated_data_good_scaled, aggregated_data_good_non_numeric], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data_preprocessed, target, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)
print("Training R-squared:", train_r2)
print("Testing R-squared:", test_r2)
