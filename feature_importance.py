import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PATH, RANDOM_STATE, TEST_SIZE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess_data


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, _ = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=5,
        bootstrap=False,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=True).tail(10)

    plt.figure(figsize=(10, 6))
    importance.plot(kind="barh")
    plt.title("Top Features Predicting Startup Success")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()