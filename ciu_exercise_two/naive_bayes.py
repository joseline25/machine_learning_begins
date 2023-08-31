import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset into a pandas DataFrame
data = pd.read_csv("individual_trash_pickup_dataset.csv") # Replace 'your_dataset.csv' with the actual filename/location of your dataset
print(data)
# Preprocess the 'Date' column
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Day'] = pd.to_datetime(data['Date']).dt.day
data['Hour'] = pd.to_datetime(data['Date']).dt.hour

# Define the features and the target variable
features = ['Year', 'Month', 'Day', 'Hour', 'Resident_ID', 'Temperature', 'Weather', 'Day_of_Week', 'Previous_Requests', 'Public_Holiday']
target = 'Resident_Trash_Pickup_Request'

X = data[features]
y = data[target]

# Perform one-hot encoding on the 'Weather' feature
encoded_X = pd.get_dummies(X, columns=['Weather', 'Day_of_Week'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2, random_state=42)

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