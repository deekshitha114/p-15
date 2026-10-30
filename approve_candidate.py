
import json
import joblib
import shutil
from pathlib import Path

METRICS = Path("drift/retraining_metrics.json")

CANDIDATE = Path(
    "drift/student_result_model_candidate.pkl"
)

APPROVED = Path("student_result_model.pkl")

MIN_ACCURACY = 0.85

if not METRICS.exists() or not CANDIDATE.exists():
    print("No candidate model available.")
    print("Existing approved model remains unchanged.")
    raise SystemExit(0)

with open(METRICS) as f:
    metrics = json.load(f)

accuracy = metrics["candidate_accuracy"]

print("Candidate accuracy:", accuracy)
print("Required accuracy:", MIN_ACCURACY)

if accuracy >= MIN_ACCURACY:

    shutil.copy2(CANDIDATE, APPROVED)

    print("QUALITY GATE: PASS")
    print("New model approved successfully.")

else:
    print("QUALITY GATE: FAIL")
    print("Old approved model is preserved.")

# Verify the approved model can be loaded.
if accuracy >= MIN_ACCURACY:
    model = joblib.load(APPROVED)
    print("Approved model loaded:", type(model).__name__)
