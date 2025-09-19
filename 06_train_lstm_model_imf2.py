
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Parametreler (BFO sonucu)
EPOCHS = 100
LEARNING_RATE = 0.01865
HIDDEN_UNITS = 103
BATCH_SIZE = 64

# Veri yükle
data = np.load("lstm_datasets/IMF_2_XY.npy")
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Eğitim/test böl
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model oluştur
model = Sequential()
model.add(LSTM(HIDDEN_UNITS, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

# Eğitim
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Tahmin
y_pred = model.predict(X_test)

# Klasör oluştur
os.makedirs("lstm_models", exist_ok=True)

# Modeli kaydet
model.save("lstm_models/IMF_2_LSTM.h5")

# Grafik
plt.figure(figsize=(10, 4))
plt.plot(y_test, label="Gerçek")
plt.plot(y_pred, label="Tahmin")
plt.title("IMF_2 LSTM Tahmin Sonucu")
plt.xlabel("Zaman")
plt.ylabel("Değer")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lstm_models/IMF_2_plot.png")
plt.show()
