import pandas as pd 
import numpy as np
import pickle 
import json
from pathlib import Path
import os 

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
y_test = np.ravel(y_test)
out = "data/predictions"

def main(repo_path):
    model_path = repo_path / "models/trained_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    metrics = {"MSE": mse, "MAE": mae, "R2": r2}
    metric_path = repo_path / "metrics/scores.json"
    
    with open(metric_path, "w") as f:
        json.dump(metrics, f)

    os.makedirs(out, exist_ok = True)

    X_test["y_true"] = y_test
    X_test["y_pred"] = predictions

    X_test.to_csv(f"{out}/prediction.csv")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)