import pandas as pd  # Import pandas for DataFrame manipulation.
import numpy as np  # Import numpy for numerical operations.
from sklearn.preprocessing import StandardScaler  # Import StandardScaler to normalize feature values.
from sklearn.model_selection import train_test_split  # Import train_test_split to divide data into training and testing sets.
from imblearn.over_sampling import SMOTE  # Import SMOTE to balance imbalanced class distributions.


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


def preprocess_dataset(df, target_column='is_fraud'):
    """
    Complete preprocessing pipeline: remove duplicates, handle missing values, and scale features.
    
    This function applies all preprocessing steps in sequence and returns a cleaned,
    normalized dataset ready for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The raw input DataFrame from the data loader.
    
    target_column : str, default='is_fraud'
        The name of the target column (should not be scaled).
    
    Returns:
    --------
    pandas.DataFrame
        The preprocessed DataFrame with:
        - Duplicates removed
        - Missing values handled
        - Numerical features scaled to mean=0, std=1
    
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler object for transforming future data.
    """
    print("=" * 60)
    print("Starting data preprocessing pipeline...")
    print("=" * 60)
    
    # Step 1: Remove duplicates.
    print("\nStep 1: Removing duplicates...")
    df_no_dupes = remove_duplicates(df)
    
    # Step 2: Handle missing values.
    print("\nStep 2: Handling missing values...")
    df_no_missing = handle_missing_values(df_no_dupes)
    
    # Step 3: Scale numerical features.
    print("\nStep 3: Scaling numerical features...")
    df_scaled, scaler = scale_features(df_no_missing, target_column=target_column)
    
    # Print final summary.
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Final dataset shape: {df_scaled.shape}")
    print("=" * 60)
    
    return df_scaled, scaler


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


def apply_smote_to_training_data(X_train, y_train, random_state=42):
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

    Returns
    -------
    X_resampled : pandas.DataFrame
        Oversampled training features as a DataFrame with the original column names.
    y_resampled : pandas.Series
        Oversampled training labels as a Series (balanced counts for each class).

    Steps
    -----
    1. Print class distribution before SMOTE so the user can see imbalance.
    2. Create and apply SMOTE to `X_train` and `y_train` only.
    3. Convert the numpy outputs back to pandas objects and print new distribution.
    4. Return balanced `X_resampled` and `y_resampled`.
    """

    # 1) Show class distribution before applying SMOTE
    # `value_counts()` gives the raw counts for each label (e.g., 0: legit, 1: fraud)
    print("Training class distribution before SMOTE:")
    counts_before = y_train.value_counts()
    print(counts_before)

    # 2) Create SMOTE instance from imbalanced-learn. SMOTE generates synthetic
    # samples of the minority class by interpolating between nearest neighbors.
    smote = SMOTE(random_state=random_state)

    # 3) Apply SMOTE to the training set only. `fit_resample` returns numpy
    # arrays for X and y. This does not touch the test set, preventing leakage.
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
    # Entry point for testing the preprocessing functions.
    from data_loader import load_creditcard_csv
    
    try:
        # Load the raw dataset.
        raw_df = load_creditcard_csv()
        print(f"Raw dataset shape: {raw_df.shape}\n")
        
        # Apply preprocessing pipeline.
        processed_df, fitted_scaler = preprocess_dataset(raw_df)
        
        # Display a preview of the processed dataset.
        print("\nFirst 5 rows of processed data:")
        print(processed_df.head())
        
        # Split the preprocessed data into training and testing sets.
        print("\n")
        X_train, X_test, y_train, y_test = split_train_test(processed_df)

        # Apply SMOTE oversampling only to the training set, leaving the test set unchanged.
        X_train_resampled, y_train_resampled = apply_smote_to_training_data(X_train, y_train)

        print("\nData split complete and ready for model training!")
        print("Note: test data is unchanged and not oversampled.")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
