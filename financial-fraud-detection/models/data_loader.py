from pathlib import Path  # Import Path to build file paths in a platform-independent way.
import pandas as pd  # Import pandas to read and analyze CSV data.


def load_creditcard_csv():
    """Load the creditcard.csv file from the project data folder."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"
    # Build the path to data/creditcard.csv relative to this script location.
    if not data_path.exists():
        # Check if the CSV file exists; raise an informative error if it doesn't.
        raise FileNotFoundError(f"CSV file not found at: {data_path}")
    return pd.read_csv(data_path)
    # Load the CSV file into a pandas DataFrame and return it.


def display_dataset_info(df):
    """Print basic dataset dimensions and fraud/legitimate counts."""
    rows, cols = df.shape
    # df.shape returns a tuple in the form (number of rows, number of columns).
    if "is_fraud" not in df.columns:
        # Check if the "is_fraud" column exists in the DataFrame; raise an error if it doesn't.
        raise KeyError(f"'is_fraud' column not found. Available columns: {list(df.columns)}")
    fraud_count = df["is_fraud"].sum()
    # Assume the target column is named "is_fraud" where 1 labels fraud and 0 labels legitimate.
    legit_count = (df["is_fraud"] == 0).sum()
    # Count rows where the "Class" column equals 0 for legitimate transactions.

    print(f"Number of rows: {rows}")
    print(f"Number of columns: {cols}")
    print(f"Fraud count: {fraud_count}")
    print(f"Legitimate count: {legit_count}")


if __name__ == "__main__":
    # Entry point: attempt to load and display dataset info.
    try:
        dataframe = load_creditcard_csv()
        # Load the dataset into a pandas DataFrame.
        display_dataset_info(dataframe)
        # Display the requested dataset metrics.
    except FileNotFoundError as e:
        # Catch the FileNotFoundError and print a helpful message to the user.
        print(f"Error: {e}")
        print("Please add creditcard.csv to the data/ folder to run this script.")
    except KeyError as e:
        # Catch KeyError if the "Class" column is missing.
        print(f"Error: {e}")
        print("Ensure creditcard.csv has a 'Class' column.")
