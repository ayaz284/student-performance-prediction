import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
data = pd.read_csv("data/student_data.csv")

print("Dataset loaded successfully!")
print("Total students:", len(data))


# Features
features = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
    "extracurricular"
]

X = data[features]
y = data["final_score"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create models
models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
}


# Train and evaluate models

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append([
        name,
        round(mae, 2),
        round(r2, 2)
    ])


# Create results table

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "R2 Score"
    ]
)


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df.to_string(index=False))


# Find best model

best_model = results_df.loc[
    results_df["R2 Score"].idxmax()
]

print("\nBest Model:")
print(best_model["Model"])

print("Best R2 Score:")
print(best_model["R2 Score"])