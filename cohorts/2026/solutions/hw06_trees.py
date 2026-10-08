"""Homework 6 — arbres, forêt aléatoire et XGBoost."""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "car_fuel_efficiency_2026.csv"
)
TARGET = "fuel_efficiency_mpg"
IMPORTANCE_CANDIDATES = [
    "vehicle_weight",
    "horsepower",
    "acceleration",
    "engine_displacement",
]


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    cars = pd.read_csv(DATA_PATH)
    features = cars.drop(columns=[TARGET]).fillna(0)
    prepared = features.copy()
    prepared[TARGET] = cars[TARGET].to_numpy()

    df_full_train, _ = train_test_split(prepared, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    y_train = df_train.pop(TARGET)
    y_val = df_val.pop(TARGET)

    encoder = DictVectorizer(sparse=True)
    X_train = encoder.fit_transform(df_train.to_dict(orient="records"))
    X_val = encoder.transform(df_val.to_dict(orient="records"))
    feature_names = list(encoder.get_feature_names_out())

    tree = DecisionTreeRegressor(max_depth=1, random_state=1)
    tree.fit(X_train, y_train)
    split_name = feature_names[int(tree.tree_.feature[0])]
    original_feature = split_name.split("=", 1)[0]

    forest_scores = {}
    for n_estimators in [10, 50, 100, 150]:
        forest = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=1,
            n_jobs=-1,
        )
        forest.fit(X_train, y_train)
        forest_scores[n_estimators] = rmse(y_val, forest.predict(X_val))
    best_n_estimators = min(forest_scores, key=forest_scores.get)

    depth_scores = {}
    for depth in [10, 15, 20, 25]:
        scores = []
        for n_estimators in [10, 50, 100, 150]:
            forest = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=depth,
                random_state=1,
                n_jobs=-1,
            )
            forest.fit(X_train, y_train)
            scores.append(rmse(y_val, forest.predict(X_val)))
        depth_scores[depth] = float(np.mean(scores))
    best_depth = min(depth_scores, key=depth_scores.get)

    importance_forest = RandomForestRegressor(
        n_estimators=10,
        max_depth=20,
        random_state=1,
        n_jobs=-1,
    )
    importance_forest.fit(X_train, y_train)
    importances = dict(zip(feature_names, importance_forest.feature_importances_))
    best_importance = max(
        IMPORTANCE_CANDIDATES,
        key=lambda name: importances.get(name, 0.0),
    )

    # XGBoost refuse certains caractères dans les noms ; on les remplace.
    safe_names = [name.replace("[", "(").replace("]", ")").replace("<", "_") for name in feature_names]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=safe_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=safe_names)
    watchlist = [(dtrain, "train"), (dval, "val")]

    eta_scores = {}
    for eta in (0.3, 0.1):
        params = {
            "eta": eta,
            "max_depth": 6,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "nthread": 8,
            "seed": 1,
            "verbosity": 0,
        }
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=watchlist,
            verbose_eval=False,
        )
        eta_scores[eta] = rmse(y_val, model.predict(dval))

    if round(eta_scores[0.3], 6) == round(eta_scores[0.1], 6):
        best_eta = "Both give equal value"
    elif eta_scores[0.3] < eta_scores[0.1]:
        best_eta = 0.3
    else:
        best_eta = 0.1

    print(f"Q1 split feature: {split_name} -> {original_feature}")
    print("Q2/Q3 RMSE by n_estimators:")
    for n_estimators, score in forest_scores.items():
        print(f"  n={n_estimators}: {score:.6f} -> {round(score, 3)}")
    print(f"Q2 n=10 option: {forest_scores[10]:.4f}")
    print(f"Q3 best n_estimators: {best_n_estimators}")
    print("Q4 mean RMSE by max_depth:")
    for depth, score in depth_scores.items():
        print(f"  depth={depth}: {score:.6f}")
    print(f"Q4 best max_depth: {best_depth}")
    print("Q5 importance:")
    for name in IMPORTANCE_CANDIDATES:
        print(f"  {name}: {importances.get(name, 0.0):.6f}")
    print(f"Q5 best feature: {best_importance}")
    print("Q6 RMSE by eta:")
    for eta, score in eta_scores.items():
        print(f"  eta={eta}: {score:.6f}")
    print(f"Q6 best eta: {best_eta}")


if __name__ == "__main__":
    main()
