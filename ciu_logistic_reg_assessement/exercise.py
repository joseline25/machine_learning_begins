import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("../messy_trash_pickup_dataset.csv")


print(data)
# 2000 rows and 8 colums


# clean data


# Get the correlation between each integer values and the label we want to predict

my_label = "Public_Holiday" # we are supposed to do this with Day_of_Week
correlations = data.corr(numeric_only=True)[my_label]
print("the different correlations are: ")
print(correlations)

# print the info on the dataset with all the columns
print(data.info())

# print some statistic on the dataset
print(data.describe())
print(data.head())

# return all the features

print("The features\n")
print(data.columns)

# return all the types

print(" The differents types\n")
print(data.dtypes)


# clean and transform data with pandas (python for data analysis chapter 7)

"""
    The DataFrame method duplicated returns a boolean Series indicating whether each
row is a duplicate or not
    """
print(data.duplicated())

#  drop_duplicates returns a DataFrame where the duplicated array is True

print(data.drop_duplicates()) # ça redevient 1000 lines
print(data.values)

# Suppose we had an additional column
# of values and wanted to filter duplicates only based on the 'k1' column

print(data.drop_duplicates(['Date']))

#duplicated and drop_duplicates by default keep the first observed value combination.
# Passing take_last=True will return the last one

data.drop_duplicates(['Date', 'Day_of_Week'], keep="last") # type: ignore

