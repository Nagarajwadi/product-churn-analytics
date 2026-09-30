import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RAW_FILE = "data/raw/raw_events.csv"
PROCESSED_FILE = "data/processed/clean_events.csv"


# --------------------------------------------------
# Extract
# --------------------------------------------------

def extract_data():
    print("Extracting raw event data...")

    df = pd.read_csv(RAW_FILE)

    print(f"Extracted {len(df):,} events.")

    return df


# --------------------------------------------------
# Transform
# --------------------------------------------------

def transform_data(df):

    print()
    print("Transforming data...")

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Remove duplicate events
    before = len(df)

    df = df.drop_duplicates()

    duplicates_removed = before - len(df)

    print(
        f"Duplicates removed: {duplicates_removed:,}"
    )

    # Sort events
    df = df.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# Validate
# --------------------------------------------------

def validate_data(df):

    print()
    print("Validating data...")

    required_columns = [
        "user_id",
        "timestamp",
        "event_name",
        "device",
        "country",
        "plan",
        "session_id",
    ]

    # Check required columns
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Check null values
    null_counts = df[required_columns].isnull().sum()

    if null_counts.sum() > 0:
        print("Null values detected:")
        print(null_counts[null_counts > 0])
    else:
        print("No null values found.")

    # Check event names
    valid_events = {
        "signup",
        "login",
        "view_product",
        "search",
        "add_to_cart",
        "purchase",
        "subscription",
        "cancel_subscription",
    }

    invalid_events = set(
        df["event_name"].unique()
    ) - valid_events

    if invalid_events:
        raise ValueError(
            f"Invalid event types: {invalid_events}"
        )

    print("Required columns: OK")
    print("Event types: OK")

    return True


# --------------------------------------------------
# Load
# --------------------------------------------------

def load_data(df):

    print()
    print("Loading processed data...")

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print(
        f"Saved {len(df):,} events to "
        f"{PROCESSED_FILE}"
    )


# --------------------------------------------------
# Main ETL pipeline
# --------------------------------------------------

def main():

    print("===================================")
    print("PRODUCT ANALYTICS ETL PIPELINE")
    print("===================================")

    # Extract
    df = extract_data()

    # Transform
    df = transform_data(df)

    # Validate
    validate_data(df)

    # Load
    load_data(df)

    print()
    print("===================================")
    print("ETL PIPELINE COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()
