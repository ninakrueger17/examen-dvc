import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import json
import numpy as np

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_train = np.ravel(y_train)


param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True]
}

rf = RandomForestRegressor(random_state=17)
gs = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring="r2")

gs.fit(X_train, y_train)

best_params = gs.best_params_

with open("./models/best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)