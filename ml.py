# download the dataset from the repo online with urllib.request
from matplotlib import pyplot as plt
import tensorflow as tf
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

# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)

"""  
Specifying a subset of a DataFrame
Pandas provide multiples ways to isolate specific rows, columns,
slices or cells in a DataFrame

"""
# print the first 3 rows

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

# print the second row
print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

# print the second to the fourth row knowing that the row
# numbers start at 0

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

# print only one colum

print("Column 'temperature':")
print(my_dataframe['temperature'])

"""
   Task 1: Create a DataFrame
Do the following:

1. Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns
    are named Eleanor, Chidi, Tahani, and Jason. Populate each of the 12 cells 
    in the DataFrame with a random integer between 0 and 100, inclusive.

2. Output the following:

    - the entire DataFrame
    - the value in the cell of row #1 of the Eleanor column
3. Create a fifth column named Janet, which is populated with the row-by-row
    sums of Tahani and Jason.

To complete this task, it helps to know the NumPy basics covered in 
the NumPy UltraQuick Tutorial. 
"""

# Create and populate a 3x4 NumPy array.
my_data = np.array([[0, 3, 4, 1], [10, 7, 20, 14], [40, 30, 15, 27]])

# Or use this method: Create a 3x4 numpy array, each cell populated with a random integer.
my_data = np.random.randint(low=0, high=101, size=(3, 4))

# Create a Python list that holds the names of the four columns.
my_column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

# Create the DataFrame .
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)

# print the row number 1 of the Eleanor column

print(my_dataframe['Eleanor'].iloc[[1]], '\n')  # type: ignore

# or
print(my_dataframe['Eleanor'][1])

# Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason
my_dataframe["Janet"] = my_dataframe["Tahani"] + my_dataframe["Jason"]


print(my_dataframe)


"""
Copying a DataFrame (optional)
Pandas provides two different ways to duplicate a DataFrame:

    - Referencing. If you assign a DataFrame to a new variable, any change to the DataFrame or to the new variable will be reflected in the other.
    - Copying. If you call the pd.DataFrame.copy method, you create a true independent copy. Changes to the original DataFrame or to the copy will not be reflected in the other.

The difference is subtle, but important.

"""

# Create a reference by assigning my_dataframe to a new variable.
print("Experiment with a reference:")
reference_to_df = my_dataframe

# Print the starting value of a particular cell.
print("  Starting value of df: %d" % my_dataframe['Jason'][1])
print("  Starting value of reference_to_df: %d\n" %
      reference_to_df['Jason'][1])

# Modify a cell in df.
my_dataframe.at[1, 'Jason'] = my_dataframe['Jason'][1] + 5
print("  Updated df: %d" % my_dataframe['Jason'][1])
print("  Updated reference_to_df: %d\n\n" % reference_to_df['Jason'][1])

# Create a true copy of my_dataframe
print("Experiment with a true copy:")
copy_of_my_dataframe = my_dataframe.copy()

# Print the starting value of a particular cell.
print("  Starting value of my_dataframe: %d" % my_dataframe['Jason'][1])
print("  Starting value of copy_of_my_dataframe: %d\n" %
      copy_of_my_dataframe['Jason'][1])

# Modify a cell in df.
my_dataframe.at[1, 'activity'] = my_dataframe['Jason'][1] + 3
print("  Updated my_dataframe: %d" % my_dataframe['Jason'][1])
print("  copy_of_my_dataframe does not get updated: %d" %
      copy_of_my_dataframe['Jason'][1])

# Linear regression with tf.keras

# Linear Regression with Synthetic Data

"""
Simple Linear Regression with Synthetic Data
In this first Colab, you'll explore linear 
regression with a simple database.


Learning objectives:
After doing this exercise, you'll know how to do the following:


    - Tune the following hyperparameters:
        . learning rate  
            (A floating-point number that tells the gradient descent
            algorithm how strongly to adjust weights and biases on each iteration.
            For example, a learning rate of 0.3 would adjust weights and biases
            three times more powerfully than a learning rate of 0.1.
            
            Learning rate is a key hyperparameter. If you set the learning rate
            too low, training will take too long. If you set the learning rate 
            too high, gradient descent often has trouble reaching convergence.)
            
        . number of epochs
        . batch size
    - Interpret different kinds of loss curves.

"""

# modules needed

# to install tensorflow ... oh my God!!!! you need a good internet connection
# le package lui même pèse 270 Mo

"""
    Define functions that build and train a model
    The following code defines two functions:

    - build_model(my_learning_rate), which builds an empty model.
    - train_model(model, feature, label, epochs), which trains the model 
        from the examples (feature and label) you pass.
        
Since you don't need to understand model building code right now,
we've hidden this code cell. You may optionally double-click the headline to 
explore this code.

"""


# Define the functions that build and train a model

def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    # A sequential model contains one or more layers.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Compile the model topography into code that
    # TensorFlow can efficiently execute. Configure
    # training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch

    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

    # Specifically gather the model's root mean
    # squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


print("Defined build_model and train_model")

"""
    Define plotting functions
    We're using a popular Python library called Matplotlib
    to create the following two plots:

    - a plot of the feature values vs. the label values, and 
        a line showing the output of the trained model.
    - a loss curve.
We hid the following code cell because learning Matplotlib
is not relevant to the learning objectives. Regardless, you must 
still run all hidden code cells.

"""

# Define the plotting functions


# @title Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against the training feature and label."""

    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")

    # Plot the feature values vs. label values.
    plt.scatter(feature, label)

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias[0]
    x1 = feature[-1]
    y1 = trained_bias[0] + (trained_weight[0][0] * x1)
    plt.plot(np.array([x0, x1]), np.array([y0, y1]), c='r')

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot the loss curve, which shows loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()


print("Defined the plot_the_model and plot_the_loss_curve functions.")

"""
    Define the dataset
The dataset consists of 12 examples. Each example consists of one feature and one label.
"""
my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,
              7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5,
            21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

"""
    Specify the hyperparameters
    The hyperparameters in this Colab are as follows:

        - learning rate
        - epochs
            (A full training pass over the entire training set such that each example
            has been processed once.

            An epoch represents N/batch size training iterations, where N is the total
            number of examples.

            For instance, suppose the following:

            The dataset consists of 1,000 examples.
            The batch size is 50 examples.
            Therefore, a single epoch requires 20 iterations:


            1 epoch = (N/batch size) = (1,000 / 50) = 20 iterations   )
            
            
        - batch_size
The following code initializes these hyperparameters and then invokes the functions
that build and train the model.

"""

learning_rate = 0.01
epochs = 10
my_batch_size = 12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
print(trained_bias)
print(trained_weight)
print(my_feature)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

"""
    Task 1: Examine the graphs
Examine the top graph. The blue dots identify the actual data; 
the red line identifies the output of the trained model. Ideally, 
the red line should align nicely with the blue dots. Does it? Probably not.

A certain amount of randomness plays into training a model, so you'll get somewhat
different results every time you train. That said, unless you are an extremely
lucky person, the red line probably doesn't align nicely with the blue dots.

Examine the bottom graph, which shows the loss curve. 

Notice that the loss curve decreases but doesn't flatten out, 
which is a sign that the model hasn't trained 
sufficiently.

"""


""" 
Task 2: Increase the number of epochs
Training loss should steadily decrease, steeply at first, and then more slowly.
Eventually, training loss should stay steady (zero slope or nearly zero slope), 
which indicates that training has converged (A state reached when loss values change
very little or not at all with each iteration.).

In Task 1, the training loss did not converge. One possible solution is to train
for more epochs. Your task is to increase the number of epochs sufficiently to get
the model to converge. However, it is inefficient to train past convergence, so don't
just set the number of epochs to an arbitrarily high value.

Examine the loss curve. Does the model converge?

"""

learning_rate=0.01
epochs= 450  # Replace ? with an integer.
my_batch_size=12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                        my_label, epochs,
                                                        my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)


"""
Task 3: Increase the learning rate
In Task 2, you increased the number of epochs to get the model to converge.
Sometimes, you can get the model to converge more quickly by increasing the learning rate.
However, setting the learning rate too high often makes it impossible for a model to 
converge. In Task 3, we've intentionally set the learning rate too high. Run the following
code cell and see what happens.

"""

# Increase the learning rate and decrease the number of epochs.
learning_rate=100 
epochs=500 

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

""" 
    The resulting model is terrible; the red line doesn't align with the blue dots.
    Furthermore, the loss curve oscillates like a roller coaster.
    An oscillating loss curve strongly suggests that the learning rate is too high.
"""

""" 
Task 4: Find the ideal combination of epochs and learning rate
Assign values to the following two hyperparameters to make training converge
as efficiently as possible:

    - learning_rate
    - epochs
"""


