
import pandas as pd
import json
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

TARGET = "result"

with open("drift/drift_report.json") as f:
    report = json.load(f)

if not report["overall_drift"]:
    print("No drift detected.")
    print("Retraining skipped.")
    raise SystemExit(0)

historical = pd.read_csv("student_results.csv")

recent = pd.read_csv(
    "drift/current_labeled_data.csv"
)

combined = pd.concat(
    [historical, recent],
    ignore_index=True
)

X = combined[FEATURES]
y = combined[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

candidate = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

candidate.fit(X_train, y_train)

predictions = candidate.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

Path("drift").mkdir(exist_ok=True)

joblib.dump(
    candidate,
    "drift/student_result_model_candidate.pkl"
)

metrics = {
    "candidate_accuracy": round(float(accuracy), 4),
    "training_rows": len(X_train),
    "test_rows": len(X_test),
    "historical_rows": len(historical),
    "recent_labeled_rows": len(recent)
}

with open("drift/retraining_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Candidate model trained successfully.")
print(json.dumps(metrics, indent=4))
