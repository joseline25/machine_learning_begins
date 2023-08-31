from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, Normalizer
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import holidays

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, precision_score, precision_recall_curve, auc, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder, MinMaxScaler

import matplotlib.pyplot as plt

data = pd.read_csv("individual_trash_pickup_dataset.csv")

"""
    1. *Data Preprocessing*:
   - Divide the dataset into two parts: one for linear regression with daily aggregated data and another for logistic regression/naive Bayes with individual resident data.
   - Handle missing values and preprocess categorical variables as needed.


"""
# display the columns of my dataset
print(data.keys())

"""
    Index(['Date', 'Resident_ID', 'Temperature', 'Weather', 'Day_of_Week',
       'Previous_Requests', 'Public_Holiday', 'Resident_Trash_Pickup_Request'],
      dtype='object')
"""

# Getting the information of the dataset
print(data.info())

# summary statistic

print(data.describe())

# plot the data dataset
data.plot(kind='scatter', x='Day_of_Week',
          y='Resident_Trash_Pickup_Request')

plt.xlabel('Day_of_Week')
plt.ylabel('Resident_Trash_Pickup_Request')
plt.show()

#  the description of the dataset
print("Description des données ")

print(data.describe())

#    check for missing values in each column
print("check for missing values")
print(data.isnull().sum())

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

print(data.info())

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
# cameroon_holidays = holidays.country_holidays('CM')
# list_cm_holidays = []
# for i in data['Date']:
#     if i in cameroon_holidays:
#         list_cm_holidays.append(1)
#     else:
#         list_cm_holidays.append(0)
# data['Public_Holiday'] = list_cm_holidays

# data['Public_Holiday'] = data['Public_Holiday'].astype('int32')


# EDA

# Investigate the distribution of daily request volumes and individual request rates
# select only the numerical columns

numeric_columns = data.select_dtypes(include=['int32'])

# plot the distribution of each numerical column
for c in numeric_columns.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=c, bins=10)
    plt.title(f'distribution of {c}')
    plt.xlabel(c)
    plt.show()

# plot the matrix of correlations

# data['Date'] = data['Date'].astype('category').cat.codes
# data['Weather'] = data['Weather'].astype('category').cat.codes
# data['Day_of_Week'] = data['Day_of_Week'].astype('category').cat.codes
# correlation_matrix = data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlations between Features')
# plt.show()


# # Normalize the dataset (scalling) with MinMaxScaler de scikit-sklearn

# # select the columns to normalize

# columns_to_normalize = ['Date', 'Resident_ID', 'Temperature', 'Previous_Requests', 'Day_of_Week']

# # create a MinMaxScaler object
# scaler = MinMaxScaler()

# # Normalize the selected colums
# data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# print(data)

# now the correlation matrix
# plot the matrix of correlations

# data['Date'] = data['Date'].astype('category').cat.codes
# data['Weather'] = data['Weather'].astype('category').cat.codes
# data['Day_of_Week'] = data['Day_of_Week'].astype('category').cat.codes
# correlation_matrix = data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlations between Features')
# plt.show()

# other plotting

# line plot for relationship between Date (then Weather, then  Day_of_Week, then Temperature)
# and Resident_Trash_Pickup_Request

plt.figure(figsize=(8, 6))
sns.lineplot(data=data, x='Previous_Requests',
             y='Resident_Trash_Pickup_Request')
plt.title('Resident_Trash_Pickup_Request over Previous_Requests')
plt.xlabel('Previous_Requests')
plt.ylabel('Resident_Trash_Pickup_Request')
plt.show()

# divide the dataset into two parts

# split the data randomly into two parts for each models

X = data[['Date', 'Resident_ID', 'Temperature', 'Weather',
          'Day_of_Week', 'Previous_Requests', 'Public_Holiday']]

y = data['Resident_Trash_Pickup_Request']

X_regr, X_class, y_regr, y_class = train_test_split(
    X, y, test_size=0.5, random_state=42)

data_reg = X_regr
data_reg['Resident_Trash_Pickup_Request'] = y_regr

data_class = X_class
data_class['Resident_Trash_Pickup_Request'] = y_class

# get the unique date
uniques_dates = data_reg['Date'].unique()

data_reg['Date'] = pd.to_datetime(data['Date']).dt.date

print(data_reg)

aggregated_data_two = data_reg.groupby(
    'Date')['Resident_Trash_Pickup_Request'].sum().reset_index()

print(aggregated_data_two)

# aggregate with all features
aggregated_data = data_reg.groupby("Date").agg({"Resident_ID": "first", "Weather": "first", "Temperature": "first",
                                                "Day_of_Week": "first", "Previous_Requests": "sum",
                                                "Public_Holiday": "first", "Resident_Trash_Pickup_Request": "sum"}).reset_index()


print(aggregated_data)

# plot date and count request
plt.figure(figsize=(8, 6))
sns.lineplot(data=aggregated_data_two, x='Date',
             y='Resident_Trash_Pickup_Request')
plt.title('Resident_Trash_Pickup_Request over Previous_Requests')
plt.xlabel('aggregated_data_two')
plt.ylabel('Resident_Trash_Pickup_Request')
plt.show()


# Normalize the dataset (scalling) with MinMaxScaler de scikit-sklearn

# select the columns to normalize

columns_to_normalize = ['Date', 'Resident_ID',
                        'Temperature', 'Previous_Requests']

# create a MinMaxScaler object
scaler = MinMaxScaler()

# Normalize the selected colums
aggregated_data['Date'] = aggregated_data['Date'].astype('category').cat.codes
data['Weather'] = data['Weather'].astype('category').cat.codes
data['Day_of_Week'] = data['Day_of_Week'].astype('category').cat.codes

aggregated_data[columns_to_normalize] = scaler.fit_transform(
    aggregated_data[columns_to_normalize])


print(aggregated_data)


# Build the linear regression model

# Select the features and target variable
my_feature = 'Date'
my_label = 'Resident_Trash_Pickup_Request'

# since I have one feature, I reshape it

x = np.array(aggregated_data[my_feature].values.tolist()).reshape((-1, 1))
y = aggregated_data[my_label].values


# Split the data into training and testing sets 70% - 30%
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

r_sq = model.score(x, y)

print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

print(f"predicted response:\n{y_pred}")
# Print the coefficients
coefficients = model.coef_
intercept = model.intercept_

print("Intercept: ", intercept)
print("Coefficients:", coefficients)

# y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response:\n{y_pred}")
print(len(y_pred.reshape((-1, 1))))
print(len(y_test))
# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs. Predicted')
plt.show()


# Plot the data points and the regression line
plt.scatter(X_test.tolist(), y_test.tolist(), color='blue', label='Data')
plt.plot(X_test.tolist(), y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

"""
Logistic Regression

"""
print(data_class)


# step 1

# split the data
features = ['Date', 'Temperature', 'Weather',
            'Day_of_Week', 'Previous_Requests', 'Public_Holiday']
target = 'Resident_Trash_Pickup_Request'

X = data_class[features]
y = data_class[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# defining our numerical columns
numeric_cols = [
    col for col in X_train.columns if X_train[col].dtypes in ['int64', 'int32']]
# defining our categorical columns
categorical_cols = [
    col for col in X_train.columns if X_train[col].dtypes == 'object']


# initializing normalizer
scaler = Normalizer(norm='l2')
# initializing encoder
encoder = OneHotEncoder(handle_unknown='ignore')

# treating numerical and categorical data
numerical_transformer = scaler
categorical_transformer = make_pipeline(encoder, scaler)


preprocessor = ColumnTransformer(transformers=[('encode', numerical_transformer, numeric_cols),
                                               ('scale', categorical_transformer, categorical_cols)])

# model initialization
model = LogisticRegression()

# initializing pipeline model
pipeline = make_pipeline(preprocessor, model)
pipeline.fit(X_train, y_train)


prediction = pipeline.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, prediction))
print("Precision: ", precision_score(y_test, prediction))
print("Recall: ", recall_score(y_test, prediction))
print("AUC-ROC: ", roc_auc_score(y_test, prediction))


# Naive Bayes

data = data_class


# Preprocess the 'Date' column
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Day'] = pd.to_datetime(data['Date']).dt.day
data['Hour'] = pd.to_datetime(data['Date']).dt.hour

# Define the features and the target variable
features = ['Year', 'Month', 'Day', 'Hour', 'Resident_ID', 'Temperature',
            'Weather', 'Day_of_Week', 'Previous_Requests', 'Public_Holiday']
target = 'Resident_Trash_Pickup_Request'

X = data[features]
y = data[target]

# Perform one-hot encoding on the 'Weather' feature
encoded_X = pd.get_dummies(X, columns=['Weather', 'Day_of_Week'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    encoded_X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
