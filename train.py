import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys

INPUT_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]

data = pd.read_csv(INPUT_PATH)
X = data.drop(columns=["diagnosed_diabetes"])
y = data["diagnosed_diabetes"]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)
joblib.dump(model, MODEL_PATH)

print("Model saved →", MODEL_PATH)
