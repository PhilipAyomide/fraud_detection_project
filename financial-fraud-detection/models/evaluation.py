import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

try:
    from .data_loader import load_creditcard_csv
    from .knn_model import DEFAULT_MODEL_FILENAME, load_model_bundle
    from .preprocessing import prepare_datasets
except ImportError:
    from data_loader import load_creditcard_csv
    from knn_model import DEFAULT_MODEL_FILENAME, load_model_bundle
    from preprocessing import prepare_datasets


def _ensure_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, output_path: Path, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if labels is not None:
        plt.xticks(np.arange(len(labels)) + 0.5, labels)
        plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_classification_report(y_true, y_pred, output_txt: Path, output_png: Path):
    report = classification_report(y_true, y_pred, digits=4)
    with open(output_txt, "w", encoding="utf-8") as report_file:
        report_file.write(report)

    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.text(0, 0.5, report, fontfamily="monospace", fontsize=10)
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def plot_roc_curve(y_true, y_score, output_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _select_model_features(X_test, model_bundle):
    """Align evaluation features with the columns used during training."""
    selected_features = model_bundle.get("selected_features") or model_bundle.get("feature_columns")
    if not selected_features:
        return X_test

    missing_features = [col for col in selected_features if col not in X_test.columns]
    if missing_features:
        raise KeyError(f"Evaluation data is missing trained features: {missing_features}")

    return X_test[selected_features]


def evaluate_model(model, X_test, y_test, output_dir=None, labels=None):
    """Run evaluation and save charts/text to `output_dir`."""
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir is None:
        output_dir = repo_root / "static" / "images"
    output_dir = Path(output_dir)
    _ensure_output_dir(output_dir)

    y_pred = model.predict(X_test)

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, cm_path, labels=labels)

    report_txt = output_dir / "classification_report.txt"
    report_png = output_dir / "classification_report.png"
    save_classification_report(y_test, y_pred, report_txt, report_png)

    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            raise AttributeError("Model has no predict_proba or decision_function")

        roc_path = output_dir / "roc_curve.png"
        plot_roc_curve(y_test, y_score, roc_path)
    except Exception as exc:
        note_path = output_dir / "roc_error.txt"
        with open(note_path, "w", encoding="utf-8") as note_file:
            note_file.write(f"ROC could not be generated: {exc}\n")


def load_model(model_path: Path):
    with open(model_path, "rb") as model_file:
        artifact = pickle.load(model_file)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]
    return artifact


def main(model_filepath: str = None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model and save charts.")
    parser.add_argument("--model", dest="model", help="Path to trained model pickle file", default=None)
    parser.add_argument("--output", dest="output", help="Output directory for images", default=None)
    args = parser.parse_args()

    raw_df = load_creditcard_csv()
    datasets = prepare_datasets(raw_df, apply_smote=False)
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]

    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "trained_models" / DEFAULT_MODEL_FILENAME
    model_path = Path(args.model) if args.model else default_model
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_bundle = load_model_bundle(model_path)
    model = model_bundle["model"]
    X_test = _select_model_features(X_test, model_bundle)

    evaluate_model(model, X_test, y_test, output_dir=(Path(args.output) if args.output else None))


if __name__ == "__main__":
    main()
