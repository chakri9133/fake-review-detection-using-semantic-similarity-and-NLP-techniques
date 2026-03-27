import json
import subprocess
import os
from config import BASE_DIR

MODELS_DIR = BASE_DIR / "All Other Models"
RESULTS_DIR = BASE_DIR / "model_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "other_models_results.json"

models = {
    "SBERT": "train_classifier.py"
}

results = {}

def run_model(model_name, script_name):
    print(f"\n🚀 Running {model_name}...\n")

    process = subprocess.run(
        ["python", str(MODELS_DIR / script_name)],
        capture_output=True,
        text=True
    )

    output = process.stdout

    # ---- Parse metrics from classification report ----
    lines = output.split("\n")
    for line in lines:
        if "accuracy" in line:
            accuracy = float(line.split()[-2])
        if line.strip().startswith("1"):
            precision = float(line.split()[1])
            recall = float(line.split()[2])
            f1 = float(line.split()[3])

    results[model_name] = {
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1
    }

    print(f"✅ {model_name} done")

# ==========================
# RUN ALL MODELS
# ==========================
for name, script in models.items():
    run_model(name, script)

# ==========================
# SAVE JSON
# ==========================
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\n==============================")
print("🎉 ALL MODEL RESULTS SAVED")
print("📄", RESULTS_PATH)
print("==============================")
