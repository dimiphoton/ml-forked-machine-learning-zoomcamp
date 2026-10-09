"""Homework 8 — réponses du réseau PyTorch à partir des historiques d'entraînement.

L'entraînement lui-même est celui de
`homework/08-deep-learning/reference_train.py`. Ce script compte les
paramètres et résume les fichiers history_*.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

HOMEWORK_DIR = Path(__file__).resolve().parents[1] / "homework" / "08-deep-learning"
sys.path.insert(0, str(HOMEWORK_DIR))

from reference_train import HairModel  # noqa: E402


def summarize(history_dir: Path) -> None:
    baseline = json.loads((history_dir / "history_baseline.json").read_text())
    augmented = json.loads((history_dir / "history_augmented.json").read_text())

    median_train_accuracy = float(np.median(baseline["train_accuracy"]))
    train_loss_std = float(np.std(baseline["train_loss"], ddof=0))
    mean_eval_loss = float(np.mean(augmented["evaluation_loss"]))
    last_five_accuracy = float(np.mean(augmented["evaluation_accuracy"][-5:]))

    print(f"Q3 median train accuracy: {median_train_accuracy:.6f} -> {round(median_train_accuracy, 2)}")
    print(f"Q4 train loss std: {train_loss_std:.6f} -> {round(train_loss_std, 3)}")
    print(f"Q5 mean evaluation loss: {mean_eval_loss:.6f} -> {round(mean_eval_loss, 3)}")
    print(
        "Q6 mean last five evaluation accuracies: "
        f"{last_five_accuracy:.6f} -> {round(last_five_accuracy, 2)}"
    )


def main() -> None:
    model = HairModel()
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print("Q1 loss: nn.BCEWithLogitsLoss()")
    print(f"Q2 trainable parameters: {n_parameters}")

    history_dir = Path(__file__).resolve().parent / "hw08_history"
    if (history_dir / "history_baseline.json").exists():
        summarize(history_dir)
    else:
        print(f"historiques absents dans {history_dir}")


if __name__ == "__main__":
    main()
