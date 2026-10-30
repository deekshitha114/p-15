
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "attendance": np.random.uniform(30, 100, n),
    "internal_marks": np.random.uniform(20, 100, n),
    "assignment_marks": np.random.uniform(20, 100, n),
    "previous_score": np.random.uniform(20, 100, n)
})

score = (
    0.30 * data["attendance"] +
    0.30 * data["internal_marks"] +
    0.20 * data["assignment_marks"] +
    0.20 * data["previous_score"]
)

data["result"] = np.where(score >= 50, "PASS", "FAIL")

data.to_csv("student_results.csv", index=False)

X = data.drop("result", axis=1)
y = data["result"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

joblib.dump(model, "student_result_model.pkl")

print("Student dataset generated successfully.")
print("Total records:", len(data))
print("Initial approved model created.")
print(data["result"].value_counts())
