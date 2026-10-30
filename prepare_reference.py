
import pandas as pd
from pathlib import Path

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

data = pd.read_csv("student_results.csv")

reference = data[FEATURES].dropna()

Path("drift").mkdir(exist_ok=True)

reference.to_csv(
    "drift/reference_data.csv",
    index=False
)

print("Reference dataset created successfully.")
print("Reference rows:", len(reference))
