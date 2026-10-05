import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_features(df):
    # Separate features and target
    X = df.drop(columns=["aqi_index"])
    y = df["aqi_index"]

    # One-hot encode categorical columns (location)
    X = pd.get_dummies(X, columns=["location"], drop_first=True)

    return X, y


def prepare_data(data_path="data/processed/aqi_cleaned.csv", test_size=0.2, random_state=42):
    # Load cleaned data
    df = pd.read_csv(data_path)

    X, y = build_features(df)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Keep dataframe format with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    print("Features ready:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
