from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../trash_pickup_dataset.csv")
print(data)  # 1000 rows and 8 features

print(list(data.keys()))  # type: ignore
# print(list(data.values))  # type: ignore
print(data.duplicated())

print(data["Day_of_Week"])

list_day_of_week = []
for i in data["Day_of_Week"]:
    match i:
        case "Monday":
            list_day_of_week.append(1)
        case "Tuesday":
            list_day_of_week.append(2)
        case "Wednesday":
            list_day_of_week.append(3)
        case "Thursday":
            list_day_of_week.append(4)
        case "Friday":
            list_day_of_week.append(5)
        case "Saturday":
            list_day_of_week.append(6)
        case _:
            list_day_of_week.append(0)


list_weather = []
for i in data["Weather"]:
    match i:
        case "Sunny":
            list_weather.append(1)
        case _:
            list_weather.append(0)


list_resident_id = []

for i in data["Resident_ID"]:
    list_resident_id.append(i)

list_date = pd.to_datetime(data["Date"])

list_previous_requests = []

for i in data["Previous_Requests"]:
    list_previous_requests.append(i)

list_public_holiday = []

for i in data["Public_Holiday"]:
    list_public_holiday.append(i)

list_trash_pickup_request = []
for i in data["Trash_Pickup_Request"]:
    list_trash_pickup_request.append(i)

list_id = [i for i in range(1000)]
my_data = np.array([list_date, list_resident_id, list_weather, list_day_of_week,
                   list_previous_requests, list_public_holiday, list_trash_pickup_request])
print(my_data.shape)

my_column_names = ['Date', 'Resident_ID', 'Weather', 'Day_of_Week',
                   'Previous_Requests', 'Public_Holiday', 'Trash_Pickup_Request']

# Create a DataFrame.
my_dataframe = pd.DataFrame(
    data=my_data.T, columns=my_column_names)  # note the .T

# Print the entire DataFrame
print(my_dataframe)
# Good !!!
print(my_dataframe.keys())


print(data)


# split the data

# X = data[['Date', 'Resident_ID', 'Weather', 'Day_of_Week',
#           'Previous_Requests', 'Public_Holiday']]

X = data[['Weather', 'Day_of_Week',
          'Previous_Requests', 'Public_Holiday']]
y = data['Trash_Pickup_Request']
# print(X)
# print(y)

# convert categorical variables into numerical representation
X = pd.get_dummies(X)

# split the data into training and testing model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# train the logistic model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions on the test set

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy", accuracy)
