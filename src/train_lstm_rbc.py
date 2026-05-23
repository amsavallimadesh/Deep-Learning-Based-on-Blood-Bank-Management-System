import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Load LSTM-ready data
# ---------------------------
X = np.load("X_rbc_lstm.npy")
y = np.load("y_rbc_lstm.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------------------------
# Train-test split (80-20)
# ---------------------------
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------
# Build LSTM model
# ---------------------------
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ---------------------------
# Train model
# ---------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------
# Evaluate model
# ---------------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MSE : {loss:.4f}")
print(f"Test MAE : {mae:.4f}")

# ---------------------------
# Save model
# ---------------------------
model.save("rbc_lstm_model.h5")
print("✅ Model saved as rbc_lstm_model.h5")

# ---------------------------
# Plot training history
# ---------------------------
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("LSTM Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
