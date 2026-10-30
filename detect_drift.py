
import pandas as pd
import json
from scipy.stats import ks_2samp
from pathlib import Path

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

reference = pd.read_csv("drift/reference_data.csv")
current = pd.read_csv("drift/current_data.csv")

results = {}
drifted_features = []

for feature in FEATURES:

    statistic, p_value = ks_2samp(
        reference[feature].dropna(),
        current[feature].dropna()
    )

    detected = p_value < 0.05

    results[feature] = {
        "ks_statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": bool(detected),
        "reference_mean": round(
            float(reference[feature].mean()), 2
        ),
        "current_mean": round(
            float(current[feature].mean()), 2
        )
    }

    if detected:
        drifted_features.append(feature)

overall_drift = len(drifted_features) >= 2

report = {
    "method": "Kolmogorov-Smirnov Test",
    "p_value_threshold": 0.05,
    "drifted_features": drifted_features,
    "drifted_feature_count": len(drifted_features),
    "overall_drift": overall_drift,
    "features": results
}

Path("drift").mkdir(exist_ok=True)

with open("drift/drift_report.json", "w") as file:
    json.dump(report, file, indent=4)

print(json.dumps(report, indent=4))

if overall_drift:
    print("DATA DRIFT DETECTED!")
else:
    print("NO SIGNIFICANT DATA DRIFT!")
