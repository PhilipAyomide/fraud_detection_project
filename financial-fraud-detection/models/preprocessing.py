import pandas as pd  # Import pandas for DataFrame manipulation.
import numpy as np  # Import numpy for numerical operations.
from sklearn.preprocessing import StandardScaler  # Import StandardScaler to normalize feature values.
from sklearn.model_selection import train_test_split  # Import train_test_split to divide data into training and testing sets.
from imblearn.over_sampling import SMOTE  # Import SMOTE to balance imbalanced class distributions.

DEFAULT_TARGET_COLUMN = "is_fraud"
IDENTIFIER_SUFFIXES = ("_id", "id")


def drop_identifier_columns(df, target_column=DEFAULT_TARGET_COLUMN):
    """Drop identifier columns that should not be used as model features."""
    feature_frame = df.drop(columns=[target_column], errors="ignore")
    identifier_cols = [
        col
        for col in feature_frame.columns
        if col.lower() == "transactionid"
        or col.lower().endswith("_id")
        or col.lower().endswith("id")
        or "identifier" in col.lower()
        or col.lower().endswith("_time")
        or col.lower() == "transaction_time"
    ]
    if identifier_cols:
        print("Identifier columns detected and dropped:")
        for col in identifier_cols:
            print(f"  - {col}")
    return df.drop(columns=identifier_cols, errors="ignore")


def encode_categorical_features(df, target_column=DEFAULT_TARGET_COLUMN, max_categories=30):
    """One-hot encode low-cardinality categorical columns for model training."""
    feature_frame = df.drop(columns=[target_column], errors="ignore")
    categorical_cols = feature_frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    if not categorical_cols:
        print("No categorical columns detected for encoding.")
        return df

    high_cardinality = [
        col for col in categorical_cols if feature_frame[col].nunique(dropna=True) > max_categories
    ]
    if high_cardinality:
        print("Dropping high-cardinality categorical columns:")
        for col in high_cardinality:
            print(f"  - {col} ({feature_frame[col].nunique(dropna=True)} unique values)")
        df = df.drop(columns=high_cardinality, errors="ignore")
        categorical_cols = [col for col in categorical_cols if col not in high_cardinality]

    if not categorical_cols:
        return df

    print("Encoding categorical columns:")
    for col in categorical_cols:
        print(f"  - {col}")

    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"Encoded feature count after one-hot encoding: {encoded.shape[1] - 1}")
    return encoded


def build_feature_matrix(df, target_column=DEFAULT_TARGET_COLUMN):
    """Clean raw data and return a feature matrix with encoded categoricals."""
    cleaned = remove_duplicates(df)
    cleaned = handle_missing_values(cleaned)
    cleaned = drop_identifier_columns(cleaned, target_column=target_column)
    categorical_cols = (
        cleaned.drop(columns=[target_column], errors="ignore")
        .select_dtypes(include=["object", "string", "category"])
        .columns.tolist()
    )
    cleaned = encode_categorical_features(cleaned, target_column=target_column)

    if target_column not in cleaned.columns:
        raise KeyError(f"Target column '{target_column}' not found after feature engineering.")

    feature_frame = cleaned.drop(columns=[target_column])
    numeric_features = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric features remain after feature engineering.")

    feature_df = cleaned[numeric_features + [target_column]].copy()
    print(f"Final training columns ({len(numeric_features)}): {numeric_features}")
    return feature_df, numeric_features, categorical_cols


def prepare_inference_sample(payload, feature_columns, scaler=None, selected_features=None):
    """Transform a raw transaction payload into model-ready features."""
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary of transaction fields.")

    frame = pd.DataFrame([payload])
    frame = drop_identifier_columns(frame, target_column=DEFAULT_TARGET_COLUMN)

    categorical_cols = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if categorical_cols:
        frame = pd.get_dummies(frame, columns=categorical_cols, drop_first=True, dtype=int)

    aligned = pd.DataFrame(0, index=[0], columns=feature_columns, dtype=float)
    for column in frame.columns:
        if column in aligned.columns:
            aligned.loc[0, column] = frame.iloc[0][column]

    if scaler is not None:
        aligned_values = scaler.transform(aligned)
        aligned = pd.DataFrame(aligned_values, columns=feature_columns)

    if selected_features:
        aligned = aligned[selected_features]

    return aligned


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


def subsample_stratified(X, y, max_samples=50000, random_state=42):
    """Reduce training size while preserving class balance for faster KNN fitting."""
    if len(X) <= max_samples:
        print(f"Training set size ({len(X)}) is within max_samples={max_samples}; no subsampling applied.")
        return X, y

    print(f"Subsampling training set from {len(X)} to {max_samples} rows (stratified).")
    X_sampled, _, y_sampled, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        stratify=y,
        random_state=random_state,
    )
    print("Subsampled class distribution:")
    print(y_sampled.value_counts())
    return X_sampled, y_sampled


def prepare_datasets(
    df,
    target_column=DEFAULT_TARGET_COLUMN,
    test_size=0.2,
    random_state=42,
    max_train_samples=50000,
    smote_sampling_strategy=0.5,
    apply_smote=True,
):
    """Canonical preprocessing pipeline without data leakage.

    Steps: clean -> encode -> split -> scale (train stats only) -> optional SMOTE (train only).
    """
    feature_df, feature_names, categorical_cols = build_feature_matrix(df, target_column=target_column)

    X_train, X_test, y_train, y_test = split_train_test(
        feature_df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, X_test, scaler = scale_train_test(X_train, X_test)
    X_train, y_train = subsample_stratified(
        X_train,
        y_train,
        max_samples=max_train_samples,
        random_state=random_state,
    )

    if apply_smote:
        print("\nApplying partial SMOTE to training data only...\n")
        X_train, y_train = apply_smote_to_training_data(
            X_train,
            y_train,
            random_state=random_state,
            sampling_strategy=smote_sampling_strategy,
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_cols,
        "target_column": target_column,
    }


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame that may contain duplicate rows.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with duplicate rows removed.
    """
    # Use drop_duplicates() to remove exact duplicate rows.
    # subset=None (default) means all columns are considered when checking for duplicates.
    df_cleaned = df.drop_duplicates(keep='first')
    # keep='first' retains the first occurrence of each duplicate row.
    
    # Print summary information about duplicates removed.
    duplicates_removed = len(df) - len(df_cleaned)
    print(f"Duplicates removed: {duplicates_removed}")
    
    return df_cleaned


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Missing values are replaced using forward fill method and then backward fill method
    to ensure all NaN values are addressed. If any NaN values remain after filling,
    they are dropped entirely.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame that may contain missing values (NaN).
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with missing values handled.
    """
    # Check the initial count of missing values per column.
    initial_missing = df.isnull().sum().sum()
    print(f"Initial missing values: {initial_missing}")
    
    # Use forward fill to propagate values forward (useful for time-series data).
    df_filled = df.ffill()
    
    # Use backward fill to handle any remaining NaN values at the beginning.
    df_filled = df_filled.bfill()
    
    # Drop any rows that still contain NaN values after filling operations.
    df_filled = df_filled.dropna()
    
    # Check the final count of missing values.
    final_missing = df_filled.isnull().sum().sum()
    rows_removed = len(df) - len(df_filled)
    print(f"Final missing values: {final_missing}")
    print(f"Rows removed due to missing values: {rows_removed}")
    
    return df_filled


def scale_features(df, target_column='is_fraud'):
    """
    Scale numerical features using StandardScaler (z-score normalization).
    
    StandardScaler transforms features to have mean=0 and standard deviation=1.
    The target column (label) is excluded from scaling as it should remain unchanged.
    Non-numeric and identifier columns are removed before scaling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with features and a target column.
    
    target_column : str, default='is_fraud'
        The name of the target column to exclude from scaling.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with scaled numerical features and the original target column.
    
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler object that can be used to transform new data later.
    """
    # Separate features and target label.
    X = df.drop(columns=[target_column])
    y = df[[target_column]]

    # Print dtypes for all columns before any feature selection or scaling.
    print("DataFrame column dtypes:")
    print(df.dtypes)

    # Detect and report non-numeric columns before scaling.
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print("Non-numeric columns detected:")
        for col in non_numeric_cols:
            print(f"  - {col}")
    else:
        print("No non-numeric columns detected.")

    identifier_cols = [
        col for col in X.columns
        if col.lower() == "transactionid"
        or col.lower().endswith("_id")
        or col.lower().endswith("id")
        or "identifier" in col.lower()
    ]

    if identifier_cols:
        print("Identifier columns detected and dropped:")
        for col in identifier_cols:
            print(f"  - {col}")
        X = X.drop(columns=identifier_cols, errors="ignore")

    if non_numeric_cols:
        # After dropping identifiers, drop any remaining non-numeric columns.
        remaining_non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if remaining_non_numeric:
            print("Dropping remaining non-numeric columns before scaling:")
            for col in remaining_non_numeric:
                print(f"  - {col}")
            X = X.drop(columns=remaining_non_numeric, errors="ignore")

    # Keep only numeric features for scaling.
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Final training columns ({len(numerical_cols)}): {numerical_cols}")

    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X[numerical_cols])
    X_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols, index=df.index)

    df_scaled = pd.concat([X_scaled_df, y], axis=1)

    print(f"Numerical features scaled: {len(numerical_cols)}")
    print("Scaling method: StandardScaler (mean=0, std=1)")

    return df_scaled, scaler


def preprocess_dataset(df, target_column=DEFAULT_TARGET_COLUMN):
    """Build a cleaned feature matrix without scaling.

    Scaling must happen after train/test split to avoid data leakage.
    Prefer `prepare_datasets()` for end-to-end model training and evaluation.
    """
    print("=" * 60)
    print("Starting feature engineering pipeline (no scaling)...")
    print("=" * 60)

    feature_df, _, _ = build_feature_matrix(df, target_column=target_column)

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print(f"Final dataset shape: {feature_df.shape}")
    print("=" * 60)

    return feature_df, None


def split_train_test(df, target_column='is_fraud', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    The data is split using a stratified approach to ensure the target variable
    distribution is consistent between train and test sets. This is important for
    imbalanced datasets like fraud detection where the fraud class is rare.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The preprocessed DataFrame to split.
    
    target_column : str, default='is_fraud'
        The name of the target column used for stratification.
    
    test_size : float, default=0.2
        The proportion of data to use for testing (20% by default).
        Training set will use 1 - test_size (80% by default).
    
    random_state : int, default=42
        Seed for the random number generator to ensure reproducibility.
    
    Returns:
    --------
    X_train : pandas.DataFrame
        Training set features (80% of the data).
    
    X_test : pandas.DataFrame
        Testing set features (20% of the data).
    
    y_train : pandas.Series
        Training set target labels (80% of the data).
    
    y_test : pandas.Series
        Testing set target labels (20% of the data).
    """
    # Separate features (X) and target (y).
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split into train and test sets using stratification.
    # stratify=y ensures that class distribution is preserved in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,           # 20% for testing
        train_size=1 - test_size,      # 80% for training (implicit)
        stratify=y,                    # Stratify by target to maintain class balance
        random_state=random_state      # Ensure reproducibility
    )
    
    # Print detailed information about the split datasets.
    print("\n" + "=" * 60)
    print("Train-Test Split Summary")
    print("=" * 60)
    print(f"Total dataset shape: {df.shape}")
    print(f"  - Rows: {df.shape[0]}")
    print(f"  - Columns: {df.shape[1]}")
    print()
    print(f"Training set shape: {X_train.shape}")
    print(f"  - Rows: {X_train.shape[0]} ({100 * X_train.shape[0] / df.shape[0]:.1f}%)")
    print(f"  - Columns: {X_train.shape[1]}")
    print(f"  - Target distribution in training set:")
    print(f"    {y_train.value_counts().to_dict()}")
    print()
    print(f"Testing set shape: {X_test.shape}")
    print(f"  - Rows: {X_test.shape[0]} ({100 * X_test.shape[0] / df.shape[0]:.1f}%)")
    print(f"  - Columns: {X_test.shape[1]}")
    print(f"  - Target distribution in testing set:")
    print(f"    {y_test.value_counts().to_dict()}")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test


def apply_smote_to_training_data(
    X_train,
    y_train,
    random_state=42,
    sampling_strategy=0.5,
):
    """Apply SMOTE oversampling to the training dataset only.

    This function MUST be called after `split_train_test()` so that SMOTE is
    applied only to the training partition (to avoid information leakage into
    the test set).

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features (only these will be oversampled).
    y_train : pandas.Series
        Training labels corresponding to `X_train`.
    random_state : int, default=42
        Random seed for SMOTE reproducibility.
    sampling_strategy : float or dict, default=0.5
        Target minority-to-majority ratio. Use 1.0 for full balancing.

    Returns
    -------
    X_resampled : pandas.DataFrame
        Oversampled training features as a DataFrame with the original column names.
    y_resampled : pandas.Series
        Oversampled training labels as a Series (balanced counts for each class).
    """

    print("Training class distribution before SMOTE:")
    counts_before = y_train.value_counts()
    print(counts_before)

    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled_array, y_resampled_array = smote.fit_resample(X_train, y_train)

    # 4) Convert back to pandas objects. Preserve original feature names.
    # The resampled data will have a new integer index; reset is intentional
    # because synthetic samples do not map to original indices.
    X_resampled = pd.DataFrame(X_resampled_array, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled_array, name=y_train.name)

    # 5) Print class distribution after SMOTE to confirm balancing.
    print("Training class distribution after SMOTE:")
    counts_after = y_resampled.value_counts()
    print(counts_after)

    # 6) Return balanced training data to be used by model training only.
    return X_resampled, y_resampled


if __name__ == "__main__":
    try:
        from data_loader import load_creditcard_csv
    except ImportError:
        from .data_loader import load_creditcard_csv

    try:
        raw_df = load_creditcard_csv()
        print(f"Raw dataset shape: {raw_df.shape}\n")

        datasets = prepare_datasets(raw_df)
        print("\nCanonical pipeline complete and ready for model training!")
        print(f"Training shape: {datasets['X_train'].shape}")
        print(f"Testing shape: {datasets['X_test'].shape}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
