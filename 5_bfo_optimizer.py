
import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Parametreler
n_fish = 10          # Daha hızlı
max_iter = 20        # Daha kısa ama yeterli
batch_size = 64

param_ranges = {
    "epochs": (100, 300),
    "learning_rate": (0.01, 0.1),
    "hidden_units": (50, 128)
}

# Komut satırından IMF adı al
if len(sys.argv) < 2:
    print("Kullanım: python 5_bfo_optimizer.py IMF_1")
    sys.exit(1)

component_name = sys.argv[1]
dataset_folder = "lstm_datasets"
result_folder = os.path.join("bfo_results", component_name)
os.makedirs(result_folder, exist_ok=True)

data_path = os.path.join(dataset_folder, f"{component_name}_XY.npy")
checkpoint_file = os.path.join(result_folder, "checkpoint.npz")

# Veri yükleme
data = np.load(data_path)
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)
X = X.reshape((X.shape[0], 1, X.shape[1]))
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Checkpoint kontrol
if os.path.exists(checkpoint_file):
    d = np.load(checkpoint_file)
    fish = d["fish"]
    fitness = d["fitness"]
    best_fish = d["best_fish"]
    best_fitness = d["best_fitness"]
    start_iter = int(d["iteration"])
    print(f"▶ [{component_name}] Devam: Iter {start_iter+1}")
else:
    fish = np.random.uniform(
        [param_ranges["epochs"][0], param_ranges["learning_rate"][0], param_ranges["hidden_units"][0]],
        [param_ranges["epochs"][1], param_ranges["learning_rate"][1], param_ranges["hidden_units"][1]],
        (n_fish, 3)
    )
    fitness = np.array([float("inf")] * n_fish)
    best_fish = None
    best_fitness = float("inf")
    start_iter = 0
    print(f"▶ [{component_name}] Yeni optimizasyon")

# Model değerlendirme fonksiyonu
def evaluate(params):
    try:
        epochs = int(params[0])
        lr = float(params[1])
        hidden = int(params[2])
        model = Sequential()
        model.add(LSTM(hidden, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, y_pred)
        return mse
    except Exception as e:
        print(f"❌ Hata: {e}")
        return float("inf")

# BFO algoritması
for t in range(start_iter, max_iter):
    SD = 1 / (1 + np.log(t + 2))
    for i in range(n_fish):
        # Yeni birey üret
        partner = np.random.randint(0, n_fish)
        if partner == i:
            partner = (i + 1) % n_fish
        step = fish[i] - fish[partner]
        rand = np.random.uniform(-1, 1, 3)
        candidate = fish[i] + SD * rand * step
        candidate = np.clip(candidate,
                            [param_ranges["epochs"][0], param_ranges["learning_rate"][0], param_ranges["hidden_units"][0]],
                            [param_ranges["epochs"][1], param_ranges["learning_rate"][1], param_ranges["hidden_units"][1]])

        candidate_fitness = evaluate(candidate)
        current_fitness = fitness[i]

        if candidate_fitness < current_fitness:
            fish[i] = candidate
            fitness[i] = candidate_fitness

        if candidate_fitness < best_fitness:
            best_fitness = candidate_fitness
            best_fish = candidate.copy()

    # Checkpoint kaydet
    np.savez(checkpoint_file,
             fish=fish,
             fitness=fitness,
             best_fish=best_fish,
             best_fitness=best_fitness,
             iteration=t + 1)

    print(f"🔁 [{component_name}] Iter {t+1}/{max_iter} → En iyi MSE: {best_fitness:.5f}")

# Final sonuç
print(f"\n🎯 [{component_name}] En iyi parametreler:")
print(f"Epochs        : {int(best_fish[0])}")
print(f"Learning Rate : {best_fish[1]:.5f}")
print(f"Hidden Units  : {int(best_fish[2])}")
print(f"MSE           : {best_fitness:.5f}")
