from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y
from pyswarms.discrete import BinaryPSO


def _binary_fitness(particles, X, y, estimator, cv, scoring, penalty):
    """Fitness function for binary particle swarm feature selection."""
    n_particles = particles.shape[0]
    costs = np.zeros(n_particles)
    for i, particle in enumerate(particles):
        mask = particle.astype(bool)
        if not np.any(mask):
            costs[i] = 1.0 + penalty
            continue

        X_subset = X[:, mask]
        estimator_clone = clone(estimator)
        try:
            scores = cross_val_score(
                estimator_clone,
                X_subset,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            score_mean = np.mean(scores)
        except Exception:
            score_mean = 0.0

        feature_ratio = mask.sum() / X.shape[1]
        costs[i] = 1.0 - score_mean + penalty * feature_ratio

    return costs


def select_pso_features(
    X,
    y,
    n_particles=30,
    iters=50,
    cv=3,
    scoring="f1",
    penalty=0.01,
    random_state=42,
    verbose=False,
):
    """Select a feature subset using binary PSO and return selected features only.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix to select from.
    y : array-like
        Target labels.
    n_particles : int, default=30
        Number of particles in the swarm.
    iters : int, default=50
        Number of PSO optimization iterations.
    cv : int, default=3
        Number of cross-validation folds used to evaluate each subset.
    scoring : str, default='f1'
        Scoring metric used by cross_val_score.
    penalty : float, default=0.01
        Penalty factor for selecting more features.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to show PSO optimization progress.

    Returns:
    -------
    X_selected : pandas.DataFrame or numpy.ndarray
        Feature matrix containing only the selected features.
    selected_features : list[str]
        Names of the selected features when X is a DataFrame.
    selected_mask : numpy.ndarray
        Boolean mask of selected features.
    best_cost : float
        Best objective cost found by PSO.
    """
    if isinstance(X, pd.DataFrame):
        identifier_cols = [
            col for col in X.columns
            if col.lower() == "transactionid"
            or col.lower().endswith("_id")
            or col.lower().endswith("id")
            or "identifier" in col.lower()
        ]
        X_numeric = X.drop(columns=identifier_cols, errors="ignore")

        non_numeric_cols = X_numeric.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            if verbose:
                print(f"Dropping non-numeric columns: {non_numeric_cols}")
            X_numeric = X_numeric.drop(columns=non_numeric_cols, errors="ignore")

        if X_numeric.shape[1] == 0:
            raise ValueError(
                "No numeric features remain after dropping identifier and non-numeric columns."
            )

        feature_names = X_numeric.columns.tolist()
        X_values = X_numeric.values
        X = X_numeric
    else:
        X_values = np.asarray(X)
        feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]

    X_values, y_values = check_X_y(X_values, y)
    dimensions = X_values.shape[1]

    estimator = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=random_state,
    )

    options = {"c1": 2.0, "c2": 2.0, "w": 0.9, "k": 15, "p": 2}
    optimizer = BinaryPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        init_pos=None,
    )

    objective = lambda particles: _binary_fitness(
        particles,
        X_values,
        y_values,
        estimator=estimator,
        cv=cv,
        scoring=scoring,
        penalty=penalty,
    )

    best_cost, best_pos = optimizer.optimize(objective, iters=iters, verbose=verbose)
    selected_mask = best_pos.astype(bool)
    selected_features = [name for name, selected in zip(feature_names, selected_mask) if selected]

    if len(selected_features) == 0:
        selected_mask[0] = True
        selected_features = [feature_names[0]]

    if isinstance(X, pd.DataFrame):
        X_selected = X.loc[:, selected_features]
    else:
        X_selected = X_values[:, selected_mask]

    return X_selected, selected_features, selected_mask, best_cost


def get_pso_selected_feature_names(X, y, **kwargs):
    """Return only the names of the features selected by PSO."""
    _, selected_features, _, _ = select_pso_features(X, y, **kwargs)
    return selected_features


if __name__ == "__main__":
    print("Loading dataset...")
    data_path = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {data_path}")

    df = pd.read_csv(data_path)

    if "Class" in df.columns:
        target_column = "Class"
    elif "is_fraud" in df.columns:
        target_column = "is_fraud"
    else:
        raise KeyError(
            f"Target column not found. Expected 'Class' or 'is_fraud'. Available columns: {df.columns.tolist()}"
        )

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print(f"Using target column: {target_column}")
    print("Running PSO...")

    X_selected, selected_features, mask, cost = select_pso_features(
        X,
        y,
        n_particles=20,
        iters=10,
        verbose=True
    )

    print("\nPSO Completed!")
    print("Best Cost:", cost)
    print("Selected Features:", selected_features)