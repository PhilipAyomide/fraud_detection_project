import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

try:
    from .preprocessing import prepare_datasets
    from .data_loader import load_creditcard_csv
    from .pso_feature_selection import select_pso_features
except ImportError:
    from preprocessing import prepare_datasets
    from data_loader import load_creditcard_csv
    from pso_feature_selection import select_pso_features

MODEL_VERSION = "1.0"
DEFAULT_MODEL_FILENAME = "baseline_best_knn.pkl"
FAST_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "algorithm": ["ball_tree"],
}


def train_knn_model(X_train, X_test, y_train, y_test, k=5):
    """Train a KNN classifier on the provided training data and return evaluation metrics."""
    model = KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_evaluation_metrics(y_test, y_pred)
    return model, metrics


def evaluate_best_model(model, X_test, y_test):
    """Compute and display the confusion matrix and classification report for the best model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print("\n" + "=" * 80)
    print("Best Model Evaluation")
    print("=" * 80)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("=" * 80)

    return cm, report


def compute_evaluation_metrics(y_test, y_pred):
    """Compute accuracy, precision, recall, and F1 for final model evaluation."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }


def perform_knn_grid_search(X_train, y_train, cv=3, param_grid=None):
    """Tune KNN hyperparameters using GridSearchCV optimizing the fraud F1 score."""
    if param_grid is None:
        param_grid = FAST_PARAM_GRID

    knn = KNeighborsClassifier(n_jobs=-1)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring="f1",
        n_jobs=-1,
        cv=cv_strategy,
        verbose=1,
        return_train_score=True,
    )

    print("\nStarting GridSearchCV for KNN hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    print("\nGridSearchCV completed.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    return grid_search


def _save_trained_model(model_bundle, filename=DEFAULT_MODEL_FILENAME):
    """Save the trained model bundle to the trained_models directory."""
    output_dir = Path(__file__).resolve().parents[1] / "trained_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with open(output_path, "wb") as model_file:
        pickle.dump(model_bundle, model_file)

    print(f"Saved trained model bundle to: {output_path}")
    return output_path


def _save_baseline_results(results_df, output_path):
    """Save the KNN comparison results to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved baseline results to: {output_path}")
    return output_path


def _apply_pso_feature_selection(X_train, y_train, X_test, verbose=False):
    """Run PSO on training data and return aligned train/test feature subsets."""
    print("\nRunning PSO feature selection on training data...")
    _, selected_features, _, best_cost = select_pso_features(
        X_train,
        y_train,
        n_particles=20,
        iters=10,
        cv=3,
        verbose=verbose,
    )
    print(f"PSO best cost: {best_cost:.4f}")
    print(f"PSO selected features ({len(selected_features)}): {selected_features}")
    return X_train[selected_features], X_test[selected_features], selected_features


def run_baseline_knn_model(
    output_path=None,
    use_pso=False,
    max_train_samples=50000,
    smote_sampling_strategy=0.5,
):
    """Run the baseline KNN workflow using the canonical preprocessing pipeline."""
    raw_df = load_creditcard_csv()

    print("\nApplying canonical preprocessing pipeline without leaking test data...")
    datasets = prepare_datasets(
        raw_df,
        max_train_samples=max_train_samples,
        smote_sampling_strategy=smote_sampling_strategy,
    )

    X_train = datasets["X_train"]
    X_test = datasets["X_test"]
    y_train = datasets["y_train"]
    y_test = datasets["y_test"]
    scaler = datasets["scaler"]
    feature_columns = datasets["feature_columns"]

    selected_features = feature_columns
    if use_pso:
        X_train, X_test, selected_features = _apply_pso_feature_selection(X_train, y_train, X_test)

    grid_search = perform_knn_grid_search(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\nEvaluating the best KNN model on the untouched test set...")
    y_pred = best_model.predict(X_test)
    metrics = compute_evaluation_metrics(y_test, y_pred)
    evaluate_best_model(best_model, X_test, y_test)

    print("\nFinal test-set metrics for best KNN model:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    results_df = pd.DataFrame(grid_search.cv_results_)[
        ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ].sort_values(by=["rank_test_score"])

    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "baseline_results.csv"

    _save_baseline_results(results_df, output_path)

    model_bundle = {
        "version": MODEL_VERSION,
        "model": best_model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "categorical_columns": datasets.get("categorical_columns", []),
        "selected_features": selected_features,
        "target_column": datasets["target_column"],
        "best_params": grid_search.best_params_,
        "metrics": metrics,
        "use_pso": use_pso,
    }
    _save_trained_model(model_bundle)

    return results_df, grid_search.best_params_, model_bundle, Path(output_path)


def load_model_bundle(model_path=None):
    """Load a saved model bundle and return the estimator plus metadata."""
    if model_path is None:
        model_path = Path(__file__).resolve().parents[1] / "trained_models" / DEFAULT_MODEL_FILENAME

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as model_file:
        artifact = pickle.load(model_file)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {
        "version": "legacy",
        "model": artifact,
        "scaler": None,
        "feature_columns": None,
        "selected_features": None,
        "target_column": "is_fraud",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the baseline KNN fraud detection model.")
    parser.add_argument("--pso", action="store_true", help="Enable PSO feature selection before training.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=50000,
        help="Maximum stratified training rows before SMOTE.",
    )
    args = parser.parse_args()

    try:
        run_baseline_knn_model(use_pso=args.pso, max_train_samples=args.max_train_samples)
    except Exception as error:
        print(f"Error while training KNN model: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
