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
from sklearn.preprocessing import StandardScaler

try:
    from .preprocessing import (
        apply_smote_to_training_data,
        handle_missing_values,
        remove_duplicates,
        split_train_test,
    )
    from .data_loader import load_creditcard_csv
except ImportError:
    from preprocessing import (
        apply_smote_to_training_data,
        handle_missing_values,
        remove_duplicates,
        split_train_test,
    )
    from data_loader import load_creditcard_csv


def train_knn_model(X_train, X_test, y_train, y_test, k=5):
    """Train a KNN classifier on the provided training data and return evaluation metrics."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

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


def test_multiple_k_values(X_train, X_test, y_train, y_test, k_values=None):
    """Tune KNN hyperparameters with GridSearchCV and evaluate the best model on the test set."""
    if k_values is None:
        k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19]

    param_grid = {
        "n_neighbors": k_values,
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        scoring="f1",
        n_jobs=-1,
        cv=cv_strategy,
        verbose=1,
        return_train_score=True,
    )

    print("\n" + "=" * 80)
    print("Tuning KNN hyperparameters with GridSearchCV")
    print("=" * 80)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = KNeighborsClassifier(**best_params)
    best_model.fit(X_train, y_train)

    print(f"\nBest parameters: {best_params}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    y_pred = best_model.predict(X_test)
    metrics = compute_evaluation_metrics(y_test, y_pred)

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(by="rank_test_score").reset_index(drop=True)

    print("\nFinal test-set metrics for best KNN model:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("=" * 80)

    return results_df, best_params, best_model


def _save_trained_model(model, filename="best_knn.pkl"):
    """Save the trained model to the trained_models directory."""
    output_dir = Path(__file__).resolve().parents[1] / "trained_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with open(output_path, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Saved trained model to: {output_path}")


def select_numeric_features(processed_df, target_column="is_fraud"):
    """Return the processed dataframe restricted to numeric feature columns and the target."""
    if target_column not in processed_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in processed dataframe.")

    feature_frame = processed_df.drop(columns=[target_column])
    numeric_features = [
        column for column in feature_frame.columns if pd.api.types.is_numeric_dtype(feature_frame[column])
    ]

    if not numeric_features:
        raise ValueError("No numeric features were found after preprocessing.")

    X = processed_df[numeric_features].copy()
    y = processed_df[target_column].copy()
    return X, y, numeric_features


def scale_train_test(X_train, X_test):
    """Fit scaler on training features and transform both train and test sets."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print("\nScaling applied after train/test split using training set statistics.")
    return X_train_scaled, X_test_scaled, scaler


def perform_knn_grid_search(X_train, y_train, cv=5):
    """Tune KNN hyperparameters using GridSearchCV optimizing the fraud F1 score."""
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2],
    }

    knn = KNeighborsClassifier()
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


def _save_baseline_results(results_df, output_path):
    """Save the KNN comparison results to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved baseline results to: {output_path}")
    return output_path


def run_baseline_knn_model(output_path=None):
    """Run the baseline KNN workflow using all processed numerical features."""
    raw_df = load_creditcard_csv()

    print("\nApplying preprocessing steps without leaking test data...")
    cleaned_df = remove_duplicates(raw_df)
    cleaned_df = handle_missing_values(cleaned_df)

    target_column = "is_fraud"
    X, y, numeric_features = select_numeric_features(cleaned_df, target_column=target_column)

    print("\nUsing all processed numerical features:")
    print(numeric_features)

    baseline_df = X.copy()
    baseline_df[target_column] = y.values

    X_train, X_test, y_train, y_test = split_train_test(
        baseline_df,
        target_column=target_column,
    )

    print("\nChecking feature scaling order...")
    X_train, X_test, scaler = scale_train_test(X_train, X_test)

    print("\nApplying SMOTE to training data only...\n")
    X_train, y_train = apply_smote_to_training_data(X_train, y_train)

    grid_search = perform_knn_grid_search(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\nEvaluating the best KNN model on the untouched test set...")
    y_pred = best_model.predict(X_test)
    metrics = compute_evaluation_metrics(y_test, y_pred)
    cm, report = evaluate_best_model(best_model, X_test, y_test)

    print("\nFinal test-set metrics for best KNN model:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "params",
        ]
    ].sort_values(by=["rank_test_score"])

    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "baseline_results.csv"

    _save_baseline_results(results_df, output_path)
    _save_trained_model(best_model, filename="baseline_best_knn.pkl")

    return results_df, grid_search.best_params_, best_model, Path(output_path)


def main():
    try:
        run_baseline_knn_model()
    except Exception as error:
        print(f"Error while training KNN model: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
    