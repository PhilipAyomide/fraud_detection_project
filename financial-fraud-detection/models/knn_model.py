import pickle
import pandas as pd
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from pso_feature_selection import select_pso_features


def train_knn_model(X_train, X_test, y_train, y_test, k=5):
    """Train a KNN classifier on the provided training data and print evaluation metrics.

    Parameters:
    -----------
    X_train : array-like or pandas.DataFrame
        Training features.
    X_test : array-like or pandas.DataFrame
        Test features.
    y_train : array-like or pandas.Series
        Training labels.
    y_test : array-like or pandas.Series
        Test labels.
    k : int, default=5
        Number of neighbors to use for KNN.

    Returns:
    -------
    model : sklearn.neighbors.KNeighborsClassifier
        The trained KNN model.
    metrics : dict
        Dictionary containing accuracy, precision, recall, and F1 score.
    """
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
    """Compute and display confusion matrix and classification report for the best model."""
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


def test_multiple_k_values(X_train, X_test, y_train, y_test, k_values=[1, 3, 5, 7, 9]):
    """Test KNN models with multiple K values and display results in a comparison table.

    Parameters:
    -----------
    X_train : array-like or pandas.DataFrame
        Training features.
    X_test : array-like or pandas.DataFrame
        Test features.
    y_train : array-like or pandas.Series
        Training labels.
    y_test : array-like or pandas.Series
        Test labels.
    k_values : list, default=[1, 3, 5, 7, 9]
        List of K values to test.

    Returns:
    -------
    results_df : pandas.DataFrame
        DataFrame containing all evaluation metrics for each K value.
    best_k : int
        The K value with the highest F1 score.
    best_model : sklearn.neighbors.KNeighborsClassifier
        The trained model with the best K value.
    """
    results = []
    best_f1 = -1.0
    best_k = None
    best_model = None

    for k in k_values:
        model, metrics = train_knn_model(X_train, X_test, y_train, y_test, k=k)
        results.append({
            "K": k,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
        })

        if metrics["f1_score"] > best_f1 or (metrics["f1_score"] == best_f1 and best_k is None):
            best_f1 = metrics["f1_score"]
            best_k = k
            best_model = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("KNN Model Performance Comparison")
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


def main():
    from preprocessing import preprocess_dataset, split_train_test, apply_smote_to_training_data
    from data_loader import load_creditcard_csv

    try:
        raw_df = load_creditcard_csv()
        processed_df, _ = preprocess_dataset(raw_df)

        print("\nSelecting PSO features from processed data...\n")
        target_column = "is_fraud"
        if target_column not in processed_df.columns:
            raise KeyError(f"Target column '{target_column}' not found in processed dataframe.")

        X = processed_df.drop(columns=[target_column])
        y = processed_df[target_column]

        selected_X, selected_features, _, _ = select_pso_features(
            X,
            y,
            n_particles=20,
            iters=10,
            verbose=False,
        )

        original_feature_count = X.shape[1]
        selected_feature_count = selected_X.shape[1]

        print("\nPSO feature selection summary")
        print("Original feature count:", original_feature_count)
        print("Selected feature count:", selected_feature_count)
        print("Selected features:", selected_features)

        X_train, X_test, y_train, y_test = split_train_test(
            pd.concat([selected_X, y], axis=1),
            target_column=target_column
        )

        # Apply SMOTE oversampling to training data only
        print("\nApplying SMOTE to training data...\n")
        X_train, y_train = apply_smote_to_training_data(X_train, y_train)

        print("\nX_train dtypes after SMOTE:")
        print(X_train.dtypes)

        print("\nTesting KNN with multiple K values...\n")
        results_df, best_k, best_model = test_multiple_k_values(
            X_train, X_test, y_train, y_test,
            k_values=[1, 3, 5, 7, 9]
        )

        print(f"\nAccuracy: {results_df.loc[results_df['K'] == best_k, 'Accuracy'].iloc[0]:.4f}")
        evaluate_best_model(best_model, X_test, y_test)
        _save_trained_model(best_model)

    except Exception as error:
        print(f"Error while training KNN model: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
