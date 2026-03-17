import pandas as pd
import joblib
from config import DATA_PATH, FINAL_MODEL_PATH, IMPUTER_PATH, RANDOM_STATE

from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, imputer = preprocess_data(df)

    final_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=5,
        bootstrap=False,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    final_model.fit(X, y)

    joblib.dump(final_model, FINAL_MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)

    print("Final model and imputer saved.")


if __name__ == "__main__":
    main()