""" 
predict price of house base on a dataset with examples having 18 features.

for this task, here is the process to create the madel.

    - get the data : the csv file in a folder 
    
"""

from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


# Import the dataset.

HOUSING_PATH = os.path.join("datasets", "exercise")


csv_path_one = os.path.join(os.getcwd(), HOUSING_PATH)
csv_path = os.path.join(csv_path_one, "housePrices_train.csv")
print(pd.read_csv(csv_path))

housing = pd.read_csv(csv_path)

# housing is a DataFrame

# print the info on the dataset with all the columns
print(housing.info())

# print some statistic on the dataset
print(housing.describe())

# Now we are going to plot the dataset with matplotlib
# plot the histogram for all numerical fields

#housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# examine the values type of the feature "SaleType"

#print(housing["SaleType"].value_counts())

#print(housing.head())
# return all the features
#print(housing.columns)
# return all the types

#print(housing.dtypes)


"""
Create a Tet set: pick aside randomly 20% of the dataset (or 
less if the dataset is too large).

"""
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


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


# @title Define the functions that build and train a model
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, df, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    history = model.fit(x=df[feature].values,
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


print("Defined the build_model and train_model functions.")

# @title Define the plotting functions


def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against 1000 random training examples."""

    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Create a scatter plot from 200 random points of the dataset.
    random_examples = housing.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).

    x0 = 0
    y0 = trained_bias[0]
    x1 = housing[feature].values[-1]

    y1 = trained_bias[0] + (trained_weight[0][0] * x1)

    plt.plot(np.array([x0, x1]), np.array([y0, y1]), c='r')

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()


print("Defined the plot_the_model and plot_the_loss_curve functions.")

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 60
batch_size = 30

# Specify the feature and the label.
my_feature = "LotArea"
my_label = "SalePrice"

# Discard any pre-existing version of the model.
my_model = None

# Invoke the functions.
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, housing,
                                         my_feature, my_label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias)


plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
