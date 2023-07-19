# download the dataset from the repo online with urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from zlib import crc32
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
import urllib.request
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# to have the housing.csv, we have call fetch_housing_data()

# Now let’s load the data using Pandas.


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()

# display the csv file with columns(attributes) and rows
# here we have 10 attributes and the method head() show the first 5 entries

print(housing.head())


"""The info() method is useful to get a 
quick description of the data, in particular the
total number of rows, and each attribute's type 
and number of non-null values """
housing.info()


# there are 20640 entries in this dataset

# get the different values of a particular field in the dataset

print(housing["households"].value_counts())
print(housing["ocean_proximity"].value_counts())

# The describe() method shows a summary of the
# numerical attributes
# le total des entrees non nulles, la moyenne, le min, le max,
# les percentiles, la deviation standard (std) qui mesure à
# quel point une valeur est dispersée

print(housing.describe())

# Now we are going to plot the dataset with matplotlib
# plot the histogram for all numerical fields

housing.hist(bins=50, figsize=(20, 15))
plt.show()

"""
Create a Tet set: pick aside randomly 20% of the dataset (or 
less if the dataset is too large).
We will use numpy for it. 
"""


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), len(test_set))

""" 
But since we are using random, everytime we will run the program,
we will have a different test set. To solve this issue, 
use each instance's identifier to decide whether or not it should go
in the test set (assuming instances have a unique and immutable identifier). For
example, you could compute a hash of each instance's identifier and put that instance
in the test set if the hash is lower or equal to 20% of the maximum hash value. This
ensures that the test set will remain consistent across multiple runs, even if you
refresh the dataset. The new test set will contain 20% of the new instances, but it will
not contain any instance that was previously in the training set. Here is a possible
implementation:
"""


def test_set_check(identifier, test_ratio):
    # type: ignore
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32 # type: ignore


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


""" 
    Unfortunately, the housing dataset does not have an identifier column. The simplest
solution is to use the row index as the ID:
"""

housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

""" 
All of this looks complicated!!!!

Scikit-Learn provides a few functions to split datasets into multiple subsets in various
ways. The simplest function is train_test_split, which does pretty much the same
thing as the function split_train_test defined earlier, with a couple of additional
features.
"""
# pip install scikit-learn
# pip install sklearn

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
plt.show()


# Pandas Colab

""" 
We are going to use Pandas DataFrame. 
For it we need numpy and pandas.

Why DataFrame ? because it is similar to an in-memory
spreadsheet. 
- store data in cells
- has named columns (usually) and numbered rows
"""

""" 
Creating a data frame of 10 cells

- 5 rows
- 2 columns : temperature and activity
"""

# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame (with pandas).
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)

"""
Adding a new column to a DataFrame
You may add a new column to an existing pandas 
DataFrame just by assigning values to a new column name. 
"""


