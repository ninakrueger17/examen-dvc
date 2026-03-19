import sklearn
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import json

print(joblib.__version__)

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

with open("./models/best_params.json", "r") as f:
    best_params = json.load(f)

best_params["max_depth"] = None if best_params["max_depth"] == "None" else best_params["max_depth"]
best_params["max_features"] = None if best_params["max_features"] == "None" else best_params["max_features"]

rf = RandomForestRegressor(**best_params, n_jobs = -1, random_state=17)

#--Train the model
rf.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/trained_model.joblib'
joblib.dump(rf, model_filename)
print("Model trained and saved successfully.")
