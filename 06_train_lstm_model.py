

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Parametreler (BFO çıktısı)
EPOCHS = 100
LEARNING_RATE = 0.09285
HIDDEN_UNITS = 73
BATCH_SIZE = 64

# Veri yükle
data = np.load("lstm_datasets/IMF_1_XY.npy")
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Eğitim/test böl
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model kur
model = Sequential()
model.add(LSTM(HIDDEN_UNITS, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

# Eğit
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Tahmin yap
y_pred = model.predict(X_test)

# Modeli kaydet
model.save("lstm_models/IMF_1_LSTM.h5")

# Grafik
plt.figure(figsize=(10, 4))
plt.plot(y_test, label="Gerçek")
plt.plot(y_pred, label="Tahmin")
plt.title("IMF_1 LSTM Tahmin Sonucu")
plt.xlabel("Zaman")
plt.ylabel("Değer")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lstm_models/IMF_1_plot.png")
plt.show()









"""
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

# 📁 Klasörler
dataset_folder = "lstm_datasets"
bfo_folder = "bfo_results"
output_folder = "predictions"
os.makedirs(output_folder, exist_ok=True)

print("🔁 LSTM eğitim süreci başlatılıyor...\n")

# 🌀 IMF_1 … IMF_10 + Residual için döngü
for i in range(1, 12):  # 1–10 IMF + 11 (Residual)
    if i == 11:
        label = "Residual"
        data_file = "Residual_XY.npy"
        checkpoint_path = os.path.join(bfo_folder, "Residual", "checkpoint.npz")
        pred_file = "y_pred_residual.npy"
    else:
        label = f"IMF_{i}"
        data_file = f"{label}_XY.npy"
        checkpoint_path = os.path.join(bfo_folder, label, "checkpoint.npz")
        pred_file = f"y_pred_{i}.npy"

    # 📥 Veri Yükle
    data_path = os.path.join(dataset_folder, data_file)
    if not os.path.exists(data_path):
        print(f"⚠ {label} için veri bulunamadı! Atlanıyor.")
        continue

    data = np.load(data_path)
    X = data[:, :-1]
    y = data[:, -1]

    # 📊 Eğitim/Test böl
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 🔁 Zaman boyutu ekle
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # ⬇ BFO çıktısı yükle
    if not os.path.exists(checkpoint_path):
        print(f"⚠ {label} için BFO sonucu yok. Atlanıyor.")
        continue

    bfo_data = np.load(checkpoint_path)
    best_fish = bfo_data["best_fish"]
    epochs = int(best_fish[0])
    lr = float(best_fish[1])
    hidden = int(best_fish[2])

    # 🧠 LSTM Modeli
    model = Sequential()
    model.add(LSTM(hidden, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

    # 🏋‍♀ Eğitim
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=0)

    # 🔮 Tahmin
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 💾 Tahminleri Kaydet
    np.save(os.path.join(output_folder, pred_file), y_pred)

    # ✅ Sonuç
    print(f"✅ {label} Eğitimi Tamamlandı:")
    print(f"   Epochs       : {epochs}")
    print(f"   LR           : {lr:.5f}")
    print(f"   Hidden Units : {hidden}")
    print(f"   MSE          : {mse:.4f}")
    print(f"   R²           : {r2:.4f}\n")

print("🎉 Tüm IMF bileşenleri başarıyla eğitildi ve tahminler kaydedildi.")
"""