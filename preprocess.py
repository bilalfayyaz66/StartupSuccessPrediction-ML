import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Create target
    df["target"] = df["status"].apply(lambda x: 1 if x == "acquired" else 0)

    # Drop leakage / noisy columns
    drop_cols = [
        "status",
        "Unnamed: 0",
        "labels",
        "id",
        "object_id",
        "name",
        "zip_code",
        "city",
        "state_code",
        "state_code.1",
        "Unnamed: 6",
        "latitude",
        "longitude",
        "founded_at",
        "first_funding_at",
        "last_funding_at",
        "closed_at",
        "age_last_milestone_year",
        "age_first_milestone_year",
        "age_last_funding_year",
        "age_first_funding_year"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

    # Feature engineering
    df["funding_per_round"] = df["funding_total_usd"] / (df["funding_rounds"] + 1)
    df["participants_per_round"] = df["avg_participants"] / (df["funding_rounds"] + 1)
    df["milestones_per_round"] = df["milestones"] / (df["funding_rounds"] + 1)
    df["network_funding_strength"] = df["relationships"] * df["avg_participants"]
    df["rounds_x_top500"] = df["funding_rounds"] * df["is_top500"]

    X = df.drop("target", axis=1)
    y = df["target"]

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y, imputer