import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filename):
    # Read data
    df = pd.read_csv(filename)

    # Remove unnessesary columns
    df.drop(['fnlwgt', 'education_num'], axis=1, inplace=True)

    # Target column becomes 0-1
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

    # One-Hot Encoding
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if 'income' in categorical_cols:
        categorical_cols.remove('income')  # Except the target

    df = pd.get_dummies(df, columns=categorical_cols)

    return df

def split_data(df):
    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    filename = "adult_combined.csv"
    df = load_and_clean_data(filename)
    print("Data successfully loaded and cleaned.")
    print(f"Total number of samples: {df.shape[0]}, total number of features: {df.shape[1]}")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
