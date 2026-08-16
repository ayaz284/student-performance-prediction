import pandas as pd
import random

# Make results reproducible
random.seed(42)

students = []

for i in range(200):

    # Generate student information
    study_hours = round(random.uniform(1, 10), 1)

    attendance = random.randint(50, 100)

    previous_score = random.randint(40, 95)

    assignments_completed = random.randint(3, 10)

    sleep_hours = round(random.uniform(5, 9), 1)

    extracurricular = random.randint(0, 1)

    # Calculate an approximate final score
    final_score = (
        study_hours * 2.5
        + attendance * 0.25
        + previous_score * 0.35
        + assignments_completed * 1.5
        + sleep_hours * 1.5
        + extracurricular * 2
    )

    # Add a small random variation
    final_score += random.uniform(-5, 5)

    # Keep score between 0 and 100
    final_score = max(0, min(100, final_score))

    students.append([
        study_hours,
        attendance,
        previous_score,
        assignments_completed,
        sleep_hours,
        extracurricular,
        round(final_score, 2)
    ])


# Create DataFrame

columns = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "sleep_hours",
    "extracurricular",
    "final_score"
]

df = pd.DataFrame(students, columns=columns)


# Save dataset

df.to_csv(
    "data/student_data.csv",
    index=False
)

print("Dataset created successfully!")
print("Number of students:", len(df))
print("Saved to: data/student_data.csv")
print()
print(df.head())