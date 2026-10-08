"""Homework 4 — métriques de classification sur le scoring de leads 2026."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "course_lead_scoring_2026.csv"
)
TARGET = "converted"
AUC_CANDIDATES = [
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
    "annual_income",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[TARGET]).copy()
    categorical = features.select_dtypes(include=["object"]).columns
    numerical = features.select_dtypes(exclude=["object"]).columns
    features[categorical] = features[categorical].fillna("NA")
    features[numerical] = features[numerical].fillna(0.0)
    features[TARGET] = df[TARGET].to_numpy()
    return features


def predict_proba(train_x: pd.DataFrame, train_y: pd.Series, val_x: pd.DataFrame, c: float):
    encoder = DictVectorizer(sparse=False)
    X_train = encoder.fit_transform(train_x.to_dict(orient="records"))
    X_val = encoder.transform(val_x.to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=c, max_iter=1000)
    model.fit(X_train, train_y)
    return model.predict_proba(X_val)[:, 1]


def fold_auc(full_train: pd.DataFrame, c: float) -> list[float]:
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for train_idx, val_idx in cv.split(full_train):
        fold_train = full_train.iloc[train_idx].copy()
        fold_val = full_train.iloc[val_idx].copy()
        y_train = fold_train.pop(TARGET)
        y_val = fold_val.pop(TARGET)
        probabilities = predict_proba(fold_train, y_train, fold_val, c=c)
        scores.append(roc_auc_score(y_val, probabilities))
    return scores


def main() -> None:
    leads = pd.read_csv(DATA_PATH)
    prepared = prepare(leads)
    numerical = [column for column in AUC_CANDIDATES if column in prepared.columns]

    df_full_train, _ = train_test_split(prepared, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    y_train = df_train.pop(TARGET)
    y_val = df_val.pop(TARGET)

    feature_auc = {}
    for column in numerical:
        score = roc_auc_score(y_train, df_train[column])
        # Un AUC sous 0.5 veut dire que la variable est inversée.
        feature_auc[column] = score if score >= 0.5 else 1.0 - score
    best_feature = max(feature_auc, key=feature_auc.get)

    probabilities = predict_proba(df_train, y_train, df_val, c=1.0)
    validation_auc = roc_auc_score(y_val, probabilities)

    thresholds = np.arange(0.0, 1.01, 0.01)
    closest = None
    f1_by_threshold = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        precision = precision_score(y_val, predicted, zero_division=0)
        recall = recall_score(y_val, predicted, zero_division=0)
        f1_value = f1_score(y_val, predicted, zero_division=0)
        f1_by_threshold.append(f1_value)
        if precision == 0 and recall == 0:
            continue
        gap = abs(precision - recall)
        if closest is None or (gap, threshold) < closest:
            closest = (gap, threshold)

    intersection = float(closest[1])
    best_f1_threshold = float(thresholds[int(np.argmax(f1_by_threshold))])

    cv_scores = fold_auc(df_full_train, c=1.0)
    cv_std = float(np.std(cv_scores))

    tuning = {}
    for c in [0.000001, 0.001, 1]:
        scores = fold_auc(df_full_train, c=c)
        tuning[c] = (float(np.mean(scores)), float(np.std(scores)))
    best_c = max(
        tuning,
        key=lambda c: (round(tuning[c][0], 3), -round(tuning[c][1], 3), -c),
    )

    print("Q1 AUC by feature:")
    for column, score in feature_auc.items():
        print(f"  {column}: {score:.6f}")
    print(f"Q1 best: {best_feature}")
    print(f"Q2 validation AUC: {validation_auc:.6f} -> {round(validation_auc, 3)}")
    print(f"Q3 precision/recall intersection: {intersection:.2f} gap={closest[0]:.6f}")
    print(f"Q4 best F1 threshold: {best_f1_threshold:.2f} f1={max(f1_by_threshold):.6f}")
    print(f"Q5 fold AUC: {[round(score, 6) for score in cv_scores]}")
    print(f"Q5 std: {cv_std:.6f} -> {round(cv_std, 3)}")
    print("Q6 mean AUC by C:")
    for c, (mean_score, std_score) in tuning.items():
        print(f"  C={c}: mean={mean_score:.6f} ({round(mean_score, 3)}) std={std_score:.6f}")
    print(f"Q6 best C: {best_c}")


if __name__ == "__main__":
    main()
