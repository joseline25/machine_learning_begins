import numpy as np
import pandas as pd


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# load data
data = pd.read_csv("../messy_trash_pickup_dataset.csv")


# handling mising values

# 1 - check for missing values in each column

print(data.isnull().sum())

# 2 - remove rows with missing values

# data = data.dropna()

# 3 - Fill missing values with a specific value

data["Weather"] = data["Weather"].fillna('Rainy')


# 4 - fill missing values with the mean or the median of the colum

data["Temperature"] = data["Temperature"].fillna(data["Temperature"].mean())


data["Previous_Requests"] = data["Previous_Requests"].fillna(
    data["Previous_Requests"].mean())

data["Public_Holiday"] = data["Public_Holiday"].fillna(
    data["Public_Holiday"].mean())

print(data.isnull().sum())


# cleaning dataù

# 1 - remove irrelevant columns

# data = data.drop("Date")

# 2 - convert date column to datetime format if it is not already

data["Date"] = pd.to_datetime(data["Date"])

# print(data["Date"])

# convert categorical variables into numerical representations
# for only one column
# data = pd.get_dummies(data, columns=["Weather"])
# for all the columns
data = pd.get_dummies(data)
# print(data.keys())

# normalize numeric features if necessary
# data["Day_of_Week"] = (data["Day_of_Week"] -
#                       data["Day_of_Week"].mean() / data["Day_of_Week"].std())


# print the first few rows of the cleaned dataset

# print(data.head())

#  predict the likelihood of a trash request on a monday

X_monday = data[['Resident_ID', 'Temperature', 'Weather_Rainy', 'Day_of_Week_Monday', 'Weather_Sunny',
                 'Previous_Requests', 'Public_Holiday']]

y = data['Trash_Pickup_Request']

# split the data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(
    X_monday, y, test_size=0.2, random_state=42)


# train the logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test set

y_pred = model.predict(X_test)

accuracy_monday = accuracy_score(y_test, y_pred)
print("Accuracy for Monday", accuracy_monday)


print("*" * 100)

#  predict the likelihood of a trash request on a tuesday

X_monday = data[['Resident_ID', 'Temperature', 'Weather_Rainy', 'Day_of_Week_Tuesday', 'Weather_Sunny',
                 'Previous_Requests', 'Public_Holiday']]

y = data['Trash_Pickup_Request']

# split the data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(
    X_monday, y, test_size=0.2, random_state=42)


# train the logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test set

y_pred = model.predict(X_test)

accuracy_tuesday = accuracy_score(y_test, y_pred)
print("Accuracy for Tuesday", accuracy_tuesday)


print("*" * 100)

#  predict the likelihood of a trash request on a wednesday

X_monday = data[['Resident_ID', 'Temperature', 'Weather_Rainy', 'Day_of_Week_Wednesday', 'Weather_Sunny',
                 'Previous_Requests', 'Public_Holiday']]

y = data['Trash_Pickup_Request']

# split the data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(
    X_monday, y, test_size=0.2, random_state=42)


# train the logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test set

y_pred = model.predict(X_test)

accuracy_wednesday = accuracy_score(y_test, y_pred)
print("Accuracy for Wednesday", accuracy_wednesday)

print("*" * 100)

#  predict the likelihood of a trash request on a thursday

X_monday = data[['Resident_ID', 'Temperature', 'Weather_Rainy', 'Day_of_Week_Thursday', 'Weather_Sunny',
                 'Previous_Requests', 'Public_Holiday']]

y = data['Trash_Pickup_Request']

# split the data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(
    X_monday, y, test_size=0.2, random_state=42)


# train the logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test set

y_pred = model.predict(X_test)

accuracy_thursday = accuracy_score(y_test, y_pred)
print("Accuracy for Thursday", accuracy_thursday)


print("*" * 100)

#  predict the likelihood of a trash request on a friday

X_monday = data[['Resident_ID', 'Temperature', 'Weather_Rainy', 'Day_of_Week_Friday', 'Weather_Sunny',
                 'Previous_Requests', 'Public_Holiday']]

y = data['Trash_Pickup_Request']

# split the data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(
    X_monday, y, test_size=0.2, random_state=42)


# train the logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test set

y_pred = model.predict(X_test)

accuracy_friday = accuracy_score(y_test, y_pred)
print("Accuracy for Friday", accuracy_friday)


print("*" * 100)

#  predict the likelihood of a trash request on a saturday

X_monday = data[['Resident_ID', 'Temperature', 'Weather_Rainy', 'Day_of_Week_Saturday', 'Weather_Sunny',
                 'Previous_Requests', 'Public_Holiday']]

y = data['Trash_Pickup_Request']

# split the data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(
    X_monday, y, test_size=0.2, random_state=42)


# train the logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test set

y_pred = model.predict(X_test)

accuracy_saturday = accuracy_score(y_test, y_pred)
print("Accuracy for Saturday", accuracy_saturday)


print("*" * 100)

#  predict the likelihood of a trash request on a sunday

X_monday = data[['Resident_ID', 'Temperature', 'Weather_Rainy', 'Day_of_Week_Sunday', 'Weather_Sunny',
                 'Previous_Requests', 'Public_Holiday']]
X_monday = normalize(X_monday, norm='max', axis=0, copy=True, return_norm=False)
print(X_monday)

y = data['Trash_Pickup_Request']

# split the data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(
    X_monday, y, test_size=0.2, random_state=42)


# train the logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test set

y_pred = model.predict(X_test)

accuracy_sunday = accuracy_score(y_test, y_pred)
print("Accuracy for Sunday", accuracy_sunday)


