"""Homework 5 — charger le pipeline figé et calculer les probabilités."""

import hashlib
import pickle
import sys
from pathlib import Path

HOMEWORK_DIR = Path(__file__).resolve().parents[1] / "homework" / "05-deployment"
sys.path.insert(0, str(HOMEWORK_DIR))

from model import normalize_record  # noqa: E402

MODEL_PATH = HOMEWORK_DIR / "pipeline.bin"
EXPECTED_SHA256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"

LEAD_Q3 = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

LEAD_Q4 = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74,
}


def conversion_probability(pipeline, lead: dict) -> float:
    record = normalize_record(lead)
    return float(pipeline.predict_proba([record])[0, 1])


def main() -> None:
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"checksum inattendu: {digest}")

    with MODEL_PATH.open("rb") as model_file:
        pipeline = pickle.load(model_file)

    probability_q3 = conversion_probability(pipeline, LEAD_Q3)
    probability_q4 = conversion_probability(pipeline, LEAD_Q4)

    dockerfile = (HOMEWORK_DIR / "Dockerfile").read_text(encoding="utf-8")
    base_image = dockerfile.splitlines()[0].removeprefix("FROM ").split("@", 1)[0]

    pyproject = (HOMEWORK_DIR / "pyproject.toml").read_text(encoding="utf-8")
    sklearn_line = next(line for line in pyproject.splitlines() if "scikit-learn" in line)

    print(f"model sha256 ok: {digest}")
    print(f"Q2 locked scikit-learn: {sklearn_line.strip()}")
    print(f"Q3 conversion probability: {probability_q3:.6f} -> {round(probability_q3, 3)}")
    print(f"Q4 conversion probability: {probability_q4:.6f} -> {round(probability_q4, 3)}")
    print(f"Q4 conversion flag: {probability_q4 >= 0.5}")
    print(f"Q5 base image: {base_image}")
    print("Q6 same probability as Q4 when the container serves this artifact")


if __name__ == "__main__":
    main()
