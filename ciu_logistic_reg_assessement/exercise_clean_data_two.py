import numpy as np
import pandas as pd


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# load data
data = pd.read_csv("../trash_pickup_dataset.csv")
print(data)

print(data.isnull().sum())

# there are no missing values


#  convert date column to datetime format if it is not already

data["Date"] = pd.to_datetime(data["Date"])



# convert categorical variables into numerical representations for all the columns

#data = pd.get_dummies(data)

print(data.keys())

print(data["Date"])

import holidays
cameroon_holidays = holidays.country_holidays('CM')
list_cm_holidays = []
for i in data["Date"]:
    if i in cameroon_holidays:
        list_cm_holidays.append(1)
    else:
        list_cm_holidays.append(0)
print(list_cm_holidays)

data["Holidays_CM"] = list_cm_holidays
print(data)


# Visualize 

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.countplot(x='Weather', data=data, hue='Trash_Pickup_Request')
plt.title("the title")
plt.xlabel("Weather conditions")
plt.ylabel('Count')
plt.show()

correlation_matrix =  data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()