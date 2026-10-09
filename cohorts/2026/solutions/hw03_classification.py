"""Homework 3 — classification logistique sur le scoring de leads 2026."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "course_lead_scoring_2026.csv"
)
TARGET = "converted"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[TARGET]).copy()
    categorical = features.select_dtypes(include=["object"]).columns
    numerical = features.select_dtypes(exclude=["object"]).columns
    features[categorical] = features[categorical].fillna("NA")
    features[numerical] = features[numerical].fillna(0.0)
    return features


def fit_logistic(train_x: pd.DataFrame, train_y: pd.Series, val_x: pd.DataFrame, c: float):
    encoder = DictVectorizer(sparse=False)
    X_train = encoder.fit_transform(train_x.to_dict(orient="records"))
    X_val = encoder.transform(val_x.to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=c, max_iter=1000, random_state=42)
    model.fit(X_train, train_y)
    return model, X_val


def main() -> None:
    leads = pd.read_csv(DATA_PATH)
    features = prepare_features(leads)
    prepared = features.copy()
    prepared[TARGET] = leads[TARGET].to_numpy()

    # Mode sur les données brutes (les NaN ne comptent pas) et après remplissage.
    raw_mode = leads["industry"].mode().iloc[0]
    filled_mode = features["industry"].mode().iloc[0]

    numerical = features.select_dtypes(exclude=["object"]).columns.tolist()
    categorical = features.select_dtypes(include=["object"]).columns.tolist()
    correlations = features[numerical].corr().abs()
    candidate_pairs = [
        ("interaction_count", "lead_score"),
        ("number_of_courses_viewed", "lead_score"),
        ("number_of_courses_viewed", "interaction_count"),
        ("annual_income", "interaction_count"),
    ]
    pair_scores = {
        pair: float(correlations.loc[pair[0], pair[1]]) for pair in candidate_pairs
    }
    best_pair = max(candidate_pairs, key=lambda pair: pair_scores[pair])

    df_full_train, df_test = train_test_split(prepared, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
    del df_test

    y_train = df_train.pop(TARGET)
    y_val = df_val.pop(TARGET)

    mutual_info = {
        column: mutual_info_score(y_train, df_train[column]) for column in categorical
    }
    best_mi = max(mutual_info, key=mutual_info.get)

    model, X_val = fit_logistic(df_train, y_train, df_val, c=1.0)
    accuracy = accuracy_score(y_val, model.predict(X_val))

    # Plus petite variation d'accuracy : la feature la moins utile.
    elimination = ["lead_source", "number_of_courses_viewed", "interaction_count"]
    signed_diff = {}
    absolute_diff = {}
    for column in elimination:
        reduced_model, reduced_X = fit_logistic(
            df_train.drop(columns=[column]),
            y_train,
            df_val.drop(columns=[column]),
            c=1.0,
        )
        reduced_accuracy = accuracy_score(y_val, reduced_model.predict(reduced_X))
        signed_diff[column] = accuracy - reduced_accuracy
        absolute_diff[column] = abs(signed_diff[column])
    least_useful = min(absolute_diff, key=absolute_diff.get)

    c_values = [0.000001, 0.00001, 0.0001, 0.001]
    c_scores = {}
    for c in c_values:
        c_model, c_X = fit_logistic(df_train, y_train, df_val, c=c)
        c_scores[c] = accuracy_score(y_val, c_model.predict(c_X))
    # En cas d'égalité, le plus petit C.
    best_c = max(c_values, key=lambda c: (round(c_scores[c], 3), -c))

    print(f"Q1 mode industry raw: {raw_mode}")
    print(f"Q1 mode industry after fill: {filled_mode}")
    print("Q2 correlations:")
    for pair, score in pair_scores.items():
        print(f"  {pair[0]} and {pair[1]}: {score:.6f}")
    print(f"Q2 best pair: {best_pair[0]} and {best_pair[1]}")
    print("Q3 mutual information:")
    for column, score in mutual_info.items():
        print(f"  {column}: {score:.6f} -> {round(score, 2)}")
    print(f"Q3 best: {best_mi}")
    print(f"Q4 accuracy: {accuracy:.6f} -> {round(accuracy, 2)}")
    print("Q5 signed difference (original - without):")
    for column, score in signed_diff.items():
        print(f"  {column}: {score:.6f} abs={absolute_diff[column]:.6f}")
    print(f"Q5 least useful (smallest absolute difference): {least_useful}")
    print("Q6 accuracy by C:")
    for c, score in c_scores.items():
        print(f"  C={c}: {score:.6f} -> {round(score, 3)}")
    print(f"Q6 best C: {best_c}")


if __name__ == "__main__":
    main()
