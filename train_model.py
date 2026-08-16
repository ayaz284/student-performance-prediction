import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
data = pd.read_csv("data/student_data.csv")

print("Dataset loaded successfully!")
print(data.head())


# Select input features
features = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
    "extracurricular"
]

X = data[features]

# Target
y = data["final_score"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# Train model
model.fit(X_train, y_train)

print("Model training completed!")


# Make predictions
predictions = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# Create model folder
os.makedirs("model", exist_ok=True)


# Save model
with open("model/student_performance_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
print("Saved as: model/student_performance_model.pkl")