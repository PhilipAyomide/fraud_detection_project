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
from sklearn.neighbors import KNeighborsClassifier

try:
    from .preprocessing import (
        apply_smote_to_training_data,
        preprocess_dataset,
        split_train_test,
    )
    from .data_loader import load_creditcard_csv
except ImportError:
    from preprocessing import (
        apply_smote_to_training_data,
        preprocess_dataset,
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


def test_multiple_k_values(X_train, X_test, y_train, y_test, k_values=None):
    """Evaluate KNN models over a range of k values and select the best one by F1 score."""
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    results = []
    best_f1 = -1.0
    best_k = None
    best_model = None

    for k in k_values:
        model, metrics = train_knn_model(X_train, X_test, y_train, y_test, k=k)
        results.append(
            {
                "K": k,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
            }
        )

        if metrics["f1_score"] > best_f1 or (metrics["f1_score"] == best_f1 and best_k is None):
            best_f1 = metrics["f1_score"]
            best_k = k
            best_model = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("KNN Baseline Performance Comparison")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    print(f"\nBest K value: {best_k} with F1 Score: {best_f1:.4f}")
    print("=" * 80)

    return results_df, int(best_k), best_model


def _save_trained_model(model, filename="best_knn.pkl"):
    """Save the trained model to the trained_models directory."""
    output_dir = Path(__file__).resolve().parents[1] / "trained_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with open(output_path, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Saved trained model to: {output_path}")


def _save_baseline_results(results_df, output_path):
    """Save the KNN comparison results to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved baseline results to: {output_path}")
    return output_path


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


def run_baseline_knn_model(output_path=None):
    """Run the baseline KNN workflow using all processed numerical features."""
    raw_df = load_creditcard_csv()
    processed_df, _ = preprocess_dataset(raw_df)

    target_column = "is_fraud"
    X, y, numeric_features = select_numeric_features(processed_df, target_column=target_column)

    print("\nUsing all processed numerical features:")
    print(numeric_features)

    baseline_df = X.copy()
    baseline_df[target_column] = y.values

    X_train, X_test, y_train, y_test = split_train_test(
        baseline_df,
        target_column=target_column,
    )

    print("\nApplying SMOTE to training data only...\n")
    X_train, y_train = apply_smote_to_training_data(X_train, y_train)

    print("\nTesting KNN with k values [1, 3, 5, 7, 9, 11, 13, 15]...\n")
    results_df, best_k, best_model = test_multiple_k_values(
        X_train,
        X_test,
        y_train,
        y_test,
        k_values=[1, 3, 5, 7, 9, 11, 13, 15],
    )

    best_row = results_df.loc[results_df["K"] == best_k].iloc[0]
    print("\nBaseline KNN summary")
    print(f"Accuracy: {best_row['Accuracy']:.4f}")
    print(f"Precision: {best_row['Precision']:.4f}")
    print(f"Recall: {best_row['Recall']:.4f}")
    print(f"F1 Score: {best_row['F1 Score']:.4f}")

    evaluate_best_model(best_model, X_test, y_test)

    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "baseline_results.csv"

    _save_baseline_results(results_df, output_path)
    _save_trained_model(best_model, filename="baseline_best_knn.pkl")

    return results_df, int(best_k), best_model, Path(output_path)


def main():
    try:
        run_baseline_knn_model()
    except Exception as error:
        print(f"Error while training KNN model: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
