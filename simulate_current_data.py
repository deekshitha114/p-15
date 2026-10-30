
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["stable", "drift"],
    default="stable"
)

args = parser.parse_args()

np.random.seed(42)

reference = pd.read_csv("drift/reference_data.csv")

current = reference.sample(
    n=300,
    replace=True,
    random_state=42
).reset_index(drop=True)

for feature in FEATURES:
    current[feature] += np.random.normal(
        0, 0.2, len(current)
    )

if args.mode == "drift":
    current["attendance"] -= 15
    current["internal_marks"] -= 10
    current["assignment_marks"] -= 8
    current["previous_score"] -= 5

for feature in FEATURES:
    current[feature] = current[feature].clip(0, 100)

Path("drift").mkdir(exist_ok=True)

current.to_csv(
    "drift/current_data.csv",
    index=False
)

# Classroom-only synthetic ground-truth labels.
# Real retraining must use actual student outcomes.
score = (
    0.30 * current["attendance"] +
    0.30 * current["internal_marks"] +
    0.20 * current["assignment_marks"] +
    0.20 * current["previous_score"]
)

labeled = current.copy()

labeled["result"] = np.where(
    score >= 50,
    "PASS",
    "FAIL"
)

labeled.to_csv(
    "drift/current_labeled_data.csv",
    index=False
)

print("Current data generated.")
print("Mode:", args.mode)
print("Number of records:", len(current))
print("Labeled recent data generated.")
