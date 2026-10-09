"""Homework 9 — graphe ONNX, prétraitement et probabilité locale."""

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ASSET_DIR = Path(os.environ.get("HW09_ASSET_DIR", "/tmp/mlz-hw09"))
SAMPLE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
MODEL_PATH = ASSET_DIR / "hair_classifier_v1.onnx"
SAMPLE_PATH = ASSET_DIR / "sample.jpeg"


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.resize((200, 200), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(array, (2, 0, 1))[None, ...]


def main() -> None:
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    image = Image.open(BytesIO(SAMPLE_PATH.read_bytes())).convert("RGB")
    tensor = prepare_image(image)
    first_red = float(tensor[0, 0, 0, 0])
    probability = float(
        session.run([output_name], {input_name: tensor})[0].reshape(-1)[0]
    )

    dockerfile = (
        Path(__file__).resolve().parents[1] / "homework" / "09-serverless" / "Dockerfile"
    ).read_text(encoding="utf-8")
    base_image = dockerfile.splitlines()[0].removeprefix("FROM ").split("@", 1)[0]

    print(f"Q1 output name: {output_name}")
    print(f"input name/shape: {input_name} {input_shape}")
    print("Q2 target size: 200x200")
    print(f"Q3 first R value: {first_red:.6f} -> {round(first_red, 3)}")
    print(f"Q4 straight probability: {probability:.6f} -> {round(probability, 3)}")
    print(f"Q4 straight flag: {probability >= 0.5}")
    print(f"Q5 base image: {base_image}")
    print("Q6 same probability as Q4 when the Lambda container uses this model")


if __name__ == "__main__":
    main()
