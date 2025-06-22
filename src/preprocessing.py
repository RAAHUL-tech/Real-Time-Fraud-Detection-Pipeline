import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess(
    input_path="data/raw/creditcard.csv",
    output_dir="data/processed/",
    drop_time=True,
    scale_amount=True,
    test_size=0.2,
    random_state=42,
    use_smote=True
):
    # Load data
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")

    # Optionally drop 'Time'
    if drop_time and "Time" in df.columns:
        df.drop(columns=["Time"], inplace=True)

    # Optionally scale 'Amount'
    if scale_amount and "Amount" in df.columns:
        scaler = StandardScaler()
        df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # Split features and labels
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified split (to maintain imbalance ratio before SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[INFO] Original train class distribution:\n{y_train.value_counts()}")
    print(f"[INFO] Test class distribution:\n{y_test.value_counts()}")

    # Apply SMOTE to training data
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[INFO] Applied SMOTE: New train class distribution:\n{y_train.value_counts()}")

    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"[INFO] Preprocessed data saved to '{output_dir}'")

if __name__ == "__main__":
    preprocess()
