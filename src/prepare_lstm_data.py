import pandas as pd
import numpy as np

WINDOW_SIZE = 14

def create_lstm_dataset(filename="scaled_rbc_demand.csv"):
    df = pd.read_csv(filename)

    # If date exists, sort and drop it
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.drop(columns=["date"])

    # ✅ Handle single-column dataset (your case)
    if df.shape[1] != 1:
        raise ValueError(
            f"❌ Expected single-column scaled dataset, found columns: {list(df.columns)}"
        )

    target_col = df.columns[0]  # -> 'rbc'
    values = df[target_col].values

    X, y = [], []

    for i in range(len(values) - WINDOW_SIZE):
        X.append(values[i:i + WINDOW_SIZE])
        y.append(values[i + WINDOW_SIZE])

    X = np.array(X).reshape(-1, WINDOW_SIZE, 1)
    y = np.array(y)

    np.save("X_rbc_lstm.npy", X)
    np.save("y_rbc_lstm.npy", y)

    print("✅ LSTM-ready data created successfully")
    print(f"Target column : {target_col}")
    print(f"X shape       : {X.shape}")
    print(f"y shape       : {y.shape}")

if __name__ == "__main__":
    create_lstm_dataset()
