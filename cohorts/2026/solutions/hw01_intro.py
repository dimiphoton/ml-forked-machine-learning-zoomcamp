"""Homework 1 — introduction : exploration du fichier carburant 2026."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "car_fuel_efficiency_2026.csv"
)


def main() -> None:
    cars = pd.read_csv(DATA_PATH)

    pandas_version = pd.__version__
    n_records = len(cars)
    n_fuel_types = cars["fuel_type"].nunique()
    n_columns_with_missing = int(cars.isna().any().sum())
    max_mpg_asia = cars.loc[cars["origin"] == "Asia", "fuel_efficiency_mpg"].max()

    horsepower = cars["horsepower"]
    median_before = horsepower.median()
    most_frequent = horsepower.mode().iloc[0]
    median_after = horsepower.fillna(most_frequent).median()
    if median_after == median_before:
        median_change = "No"
    elif median_after > median_before:
        median_change = "Yes, it increased"
    else:
        median_change = "Yes, it decreased"

    # Les 7 premières lignes Asie, dans l'ordre du fichier.
    asia = cars.loc[cars["origin"] == "Asia", ["vehicle_weight", "model_year"]].head(7)
    x = asia.to_numpy()
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    xtx = x.T @ x
    w = np.linalg.inv(xtx) @ x.T @ y
    weight_sum = float(w.sum())

    print(f"Q1 pandas version: {pandas_version}")
    print(f"Q2 records: {n_records}")
    print(f"Q3 fuel types: {n_fuel_types}")
    print(f"Q4 columns with missing values: {n_columns_with_missing}")
    print(f"Q5 max fuel efficiency Asia: {max_mpg_asia:.1f}")
    print(f"Q6 median before: {median_before}")
    print(f"Q6 most frequent horsepower: {most_frequent}")
    print(f"Q6 median after fill: {median_after}")
    print(f"Q6 change: {median_change}")
    print(f"Q7 sum of weights: {weight_sum:.4f}")


if __name__ == "__main__":
    main()
