"""Flask application entry point for the fraud detection MVP."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from knn_model import DEFAULT_MODEL_FILENAME, load_model_bundle
from preprocessing import prepare_inference_sample

app = Flask(__name__)
_model_bundle = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        model_path = PROJECT_ROOT / "trained_models" / DEFAULT_MODEL_FILENAME
        _model_bundle = load_model_bundle(model_path)
    return _model_bundle


def _prepare_prediction_frame(payload: dict, model_bundle: dict) -> pd.DataFrame:
    """Transform a single API payload into the feature frame expected by the model."""
    feature_columns = model_bundle.get("feature_columns")
    selected_features = model_bundle.get("selected_features") or feature_columns
    scaler = model_bundle.get("scaler")

    if not feature_columns:
        raise ValueError("Saved model is missing feature metadata. Retrain with the latest pipeline.")

    return prepare_inference_sample(
        payload,
        feature_columns=feature_columns,
        scaler=scaler,
        selected_features=selected_features,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "financial-fraud-detection"})


@app.route("/model-info", methods=["GET"])
def model_info():
    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(
        {
            "version": bundle.get("version"),
            "target_column": bundle.get("target_column"),
            "use_pso": bundle.get("use_pso", False),
            "feature_count": len(bundle.get("feature_columns") or []),
            "selected_feature_count": len(bundle.get("selected_features") or []),
            "best_params": bundle.get("best_params"),
            "metrics": bundle.get("metrics"),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        feature_frame = _prepare_prediction_frame(payload, bundle)
        model = bundle["model"]
        prediction = int(model.predict(feature_frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(feature_frame)[0][1])

        return jsonify(
            {
                "is_fraud": prediction,
                "fraud_probability": probability,
                "label": "fraud" if prediction == 1 else "legitimate",
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
