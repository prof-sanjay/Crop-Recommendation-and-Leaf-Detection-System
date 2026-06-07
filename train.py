import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle

# Load dataset
PATH = 'Crop_recommendation.csv'
df = pd.read_csv(PATH)

# Separate features and target label
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# Initialize and train the Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=20, random_state=0)
random_forest_model.fit(X_train, y_train)

# Making predictions
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest's Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save the trained Random Forest model
rf_model_filename = 'RandomForest.pkl'
with open(rf_model_filename, 'wb') as file:
    pickle.dump(random_forest_model, file)

# Making a prediction with the trained model
data_sample_1 = np.array([[104, 18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction_1 = random_forest_model.predict(data_sample_1)
print(f"Prediction for data sample 1: {prediction_1}")

data_sample_2 = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction_2 = random_forest_model.predict(data_sample_2)
print(f"Prediction for data sample 2: {prediction_2}")
