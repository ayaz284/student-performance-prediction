import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

with open("model/student_performance_model.pkl", "rb") as file:
    model = pickle.load(file)


# Load dataset
dataset = pd.read_csv("data/student_data.csv")


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 Student Performance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Academic Performance Prediction System</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📌 About Project")

    st.write(
        """
        This application uses Machine Learning
        to predict a student's expected final score.

        The model considers:

        • Study hours
        • Attendance
        • Previous score
        • Assignments
        • Sleep hours
        • Extracurricular activity
        """
    )

    st.divider()

    st.write("👨‍💻 Developed by:")
    st.write("**Shaik Ayaz**")

    st.write("🎓 B.Tech Artificial Intelligence")


# --------------------------------------------------
# STUDENT INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📝 Student Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    study_hours = st.number_input(
        "📚 Daily Study Hours",
        min_value=0.0,
        max_value=15.0,
        value=4.0,
        step=0.5
    )

    attendance = st.number_input(
        "📅 Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    previous_score = st.number_input(
        "📊 Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=1.0
    )


with col2:

    assignments_completed = st.number_input(
        "📋 Assignments Completed",
        min_value=0,
        max_value=20,
        value=7,
        step=1
    )

    sleep_hours = st.number_input(
        "😴 Daily Sleep Hours",
        min_value=0.0,
        max_value=15.0,
        value=7.0,
        step=0.5
    )

    extracurricular = st.selectbox(
        "🏆 Extracurricular Activities",
        ["No", "Yes"]
    )


# Convert Yes/No into 0/1

extracurricular_value = (
    1 if extracurricular == "Yes" else 0
)


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

predict_button = st.button(
    "🔮 Predict Student Performance",
    use_container_width=True
)


if predict_button:

    input_data = pd.DataFrame({

        "study_hours": [study_hours],

        "attendance": [attendance],

        "previous_score": [previous_score],

        "assignments_completed": [assignments_completed],

        "sleep_hours": [sleep_hours],

        "extracurricular": [extracurricular_value]

    })


    # Prediction

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))


    # Performance category

    if prediction >= 90:

        category = "Excellent 🌟"

    elif prediction >= 75:

        category = "Very Good 👍"

    elif prediction >= 60:

        category = "Good 🙂"

    elif prediction >= 50:

        category = "Average"

    else:

        category = "Needs Improvement"


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.header("📈 Prediction Result")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Predicted Final Score",
            f"{prediction:.2f}%"
        )


    with result_col2:

        st.metric(
            "Performance Level",
            category
        )


    st.write("### Performance Score")

    st.progress(
        int(prediction)
    )


    # Student summary

    st.write("### 📋 Student Summary")


    summary = pd.DataFrame({

        "Parameter": [

            "Study Hours",

            "Attendance",

            "Previous Score",

            "Assignments",

            "Sleep Hours",

            "Extracurricular"

        ],

        "Value": [

            f"{study_hours} hours/day",

            f"{attendance}%",

            f"{previous_score}%",

            assignments_completed,

            f"{sleep_hours} hours/day",

            extracurricular

        ]

    })


    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )


# --------------------------------------------------
# ANALYTICS SECTION
# --------------------------------------------------

st.divider()

st.header("📊 Student Performance Analytics")

st.write(
    "Explore the relationship between student factors and final performance."
)


# Dataset statistics

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Students",
        len(dataset)
    )


with col2:

    st.metric(
        "Average Final Score",
        f"{dataset['final_score'].mean():.2f}%"
    )


with col3:

    st.metric(
        "Average Attendance",
        f"{dataset['attendance'].mean():.2f}%"
    )


# --------------------------------------------------
# STUDY HOURS CHART
# --------------------------------------------------

st.subheader("📚 Study Hours vs Final Score")

fig1, ax1 = plt.subplots()

ax1.scatter(
    dataset["study_hours"],
    dataset["final_score"]
)

ax1.set_xlabel("Study Hours")

ax1.set_ylabel("Final Score")

ax1.set_title(
    "Study Hours vs Final Score"
)

st.pyplot(fig1)


# --------------------------------------------------
# ATTENDANCE CHART
# --------------------------------------------------

st.subheader("📅 Attendance vs Final Score")

fig2, ax2 = plt.subplots()

ax2.scatter(
    dataset["attendance"],
    dataset["final_score"]
)

ax2.set_xlabel("Attendance (%)")

ax2.set_ylabel("Final Score")

ax2.set_title(
    "Attendance vs Final Score"
)

st.pyplot(fig2)


# --------------------------------------------------
# PREVIOUS SCORE CHART
# --------------------------------------------------

st.subheader("📊 Previous Score vs Final Score")

fig3, ax3 = plt.subplots()

ax3.scatter(
    dataset["previous_score"],
    dataset["final_score"]
)

ax3.set_xlabel("Previous Score")

ax3.set_ylabel("Final Score")

ax3.set_title(
    "Previous Score vs Final Score"
)

st.pyplot(fig3)


# --------------------------------------------------
# --------------------------------------------------
# AUTOMATIC MODEL COMPARISON
# --------------------------------------------------

st.divider()

st.header("🤖 Machine Learning Model Comparison")

st.write(
    "The following models are trained and evaluated automatically "
    "using the current student dataset."
)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Features and target

features = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
    "extracurricular"
]

X = dataset[features]
y = dataset["final_score"]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Models

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}


# Evaluate models

results = []

for name, ml_model in models.items():

    ml_model.fit(X_train, y_train)

    predictions = ml_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "R2 Score": round(r2, 2)
    })


# Results table

model_results = pd.DataFrame(results)


st.dataframe(
    model_results,
    use_container_width=True,
    hide_index=True
)


# Find best model

best_index = model_results["R2 Score"].idxmax()

best_model = model_results.loc[
    best_index,
    "Model"
]

best_score = model_results.loc[
    best_index,
    "R2 Score"
]


st.success(
    f"🏆 Best Model: **{best_model}** "
    f"with an R² Score of **{best_score:.2f}**"
)


# Chart

st.subheader("📊 Model R² Score Comparison")

fig4, ax4 = plt.subplots()

ax4.bar(
    model_results["Model"],
    model_results["R2 Score"]
)

ax4.set_xlabel("Machine Learning Model")

ax4.set_ylabel("R² Score")

ax4.set_title(
    "Machine Learning Model Performance"
)

ax4.set_ylim(0, 1)

st.pyplot(fig4)

st.divider()

st.header("🤖 Machine Learning Model Comparison")

st.write(
    "Three regression models were evaluated using "
    "Mean Absolute Error (MAE) and R² Score."
)


# Model comparison results

model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        2.41,
        5.44,
        4.54
    ],

    "R2 Score": [
        0.92,
        0.62,
        0.74
    ]
})


# Display table

st.dataframe(
    model_results,
    use_container_width=True,
    hide_index=True
)


# Find best model

best_model = model_results.loc[
    model_results["R2 Score"].idxmax(),
    "Model"
]

best_score = model_results["R2 Score"].max()


st.success(
    f"🏆 Best Model: **{best_model}** "
    f"with an R² Score of **{best_score:.2f}**"
)


# R2 comparison chart

st.subheader("📊 R² Score Comparison")

fig4, ax4 = plt.subplots()

ax4.bar(
    model_results["Model"],
    model_results["R2 Score"]
)

ax4.set_ylabel("R² Score")

ax4.set_xlabel("Machine Learning Model")

ax4.set_title(
    "Model Performance Comparison"
)

ax4.set_ylim(0, 1)

st.pyplot(fig4)
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Performance Prediction | Machine Learning Project | Developed by Shaik Ayaz"
)