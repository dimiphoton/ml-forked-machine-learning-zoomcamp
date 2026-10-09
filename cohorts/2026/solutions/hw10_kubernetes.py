"""Homework 10 — réponses lues dans les manifests Kubernetes 2026.

La probabilité de la question 1 est celle du même lead que le devoir 5,
servi par l'image `zoomcamp-model:2026-hw10`.
"""

import hashlib
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = ROOT / "homework" / "05-deployment"
K8S_DIR = ROOT / "homework" / "10-kubernetes"
sys.path.insert(0, str(DEPLOYMENT_DIR))

from model import normalize_record  # noqa: E402

LEAD = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74,
}


def field_after(text: str, key: str) -> str:
    """Lit une valeur YAML simple, y compris derrière un tiret de liste."""
    needle = key if key.endswith(":") else f"{key}:"
    for line in text.splitlines():
        stripped = line.strip().lstrip("- ").strip()
        if stripped.startswith(needle):
            return stripped.split(":", 1)[1].strip()
    raise KeyError(key)


def main() -> None:
    model_path = DEPLOYMENT_DIR / "pipeline.bin"
    with model_path.open("rb") as model_file:
        pipeline = pickle.load(model_file)
    record = normalize_record(LEAD)
    probability = float(pipeline.predict_proba([record])[0, 1])

    deployment = (K8S_DIR / "deployment.yaml").read_text(encoding="utf-8")
    service = (K8S_DIR / "service.yaml").read_text(encoding="utf-8")
    hpa = (K8S_DIR / "hpa.yaml").read_text(encoding="utf-8")

    print(f"Q1 conversion probability: {probability:.6f} -> {round(probability, 3)}")
    print("Q3 smallest unit: Pod")
    print("Q4 default kubernetes service TYPE on kind: ClusterIP")
    print("Q5 image load command: kind load docker-image")
    print(f"Q6 containerPort: {field_after(deployment, 'containerPort:')}")
    print(f"Q7 selector app: {field_after(service, 'app:')}")
    print(f"Q8 maxReplicas: {field_after(hpa, 'maxReplicas:')}")
    print("model sha256:", hashlib.sha256(model_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
