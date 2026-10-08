"""Homework 2 — régression linéaire sur la consommation (MPG)."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "car_fuel_efficiency_2026.csv"
)
FEATURES = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
]
TARGET = "fuel_efficiency_mpg"


def split_like_lecture(df: pd.DataFrame, seed: int):
    """Découpage 60/20/20 du cours : seed NumPy puis shuffle."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train : n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val :]]
    return df_train, df_val, df_test


def train_linear_regression(X: np.ndarray, y: np.ndarray, r: float = 0.0):
    """Équation normale du cours. r > 0 régularise aussi le biais."""
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = y_pred - y_true
    return float(np.sqrt((error**2).mean()))


def prepare_X(df: pd.DataFrame, fill_value: float) -> np.ndarray:
    return df[FEATURES].fillna(fill_value).to_numpy(dtype=float)


def score_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    fill_value: float,
    r: float = 0.0,
) -> float:
    X_train = prepare_X(df_train, fill_value)
    y_train = df_train[TARGET].to_numpy(dtype=float)
    w0, w = train_linear_regression(X_train, y_train, r=r)

    X_val = prepare_X(df_val, fill_value)
    y_val = df_val[TARGET].to_numpy(dtype=float)
    y_pred = w0 + X_val.dot(w)
    return rmse(y_val, y_pred)


def main() -> None:
    cars = pd.read_csv(DATA_PATH)
    data = cars[FEATURES + [TARGET]].copy()

    missing_columns = data.columns[data.isna().any()].tolist()
    horsepower_median = float(data["horsepower"].median())

    df_train, df_val, df_test = split_like_lecture(data, seed=42)
    train_mean = float(df_train["horsepower"].mean())
    rmse_zero = score_model(df_train, df_val, fill_value=0.0)
    rmse_mean = score_model(df_train, df_val, fill_value=train_mean)

    if round(rmse_zero, 3) == round(rmse_mean, 3):
        better_imputation = "Both are equally good"
    elif rmse_zero < rmse_mean:
        better_imputation = "With 0"
    else:
        better_imputation = "With mean"

    regularization_values = [0, 0.01, 0.1, 1, 5, 10, 100]
    regularization_scores = {
        r: score_model(df_train, df_val, fill_value=0.0, r=r)
        for r in regularization_values
    }
    best_r = min(
        regularization_values,
        key=lambda r: (round(regularization_scores[r], 4), r),
    )

    seed_scores = []
    for seed in range(10):
        seed_train, seed_val, _ = split_like_lecture(data, seed=seed)
        seed_scores.append(score_model(seed_train, seed_val, fill_value=0.0))
    score_std = float(np.std(seed_scores))

    train_9, val_9, test_9 = split_like_lecture(data, seed=9)
    combined = pd.concat([train_9, val_9])
    test_rmse = score_model(combined, test_9, fill_value=0.0, r=0.001)

    print(f"Q1 missing column: {missing_columns[0]}")
    print(f"Q2 horsepower median: {horsepower_median:.0f}")
    print(f"Q3 RMSE fill 0: {rmse_zero:.6f} -> {round(rmse_zero, 3)}")
    print(f"Q3 RMSE fill mean: {rmse_mean:.6f} -> {round(rmse_mean, 3)}")
    print(f"Q3 better option: {better_imputation}")
    print("Q4 RMSE by r:")
    for r in regularization_values:
        print(f"  r={r}: {regularization_scores[r]:.6f} -> {round(regularization_scores[r], 4)}")
    print(f"Q4 best r: {best_r}")
    print(f"Q5 validation RMSE by seed: {[round(score, 6) for score in seed_scores]}")
    print(f"Q5 std: {score_std:.6f} -> {round(score_std, 3)}")
    print(f"Q6 test RMSE: {test_rmse:.6f} -> {round(test_rmse, 3)}")


if __name__ == "__main__":
    main()
