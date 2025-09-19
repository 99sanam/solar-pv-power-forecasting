import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 📥 Dosya yolları
y_test = np.load("y_test.npy")
y_pred = np.load("predictions/y_pred_final.npy")

# ✂ Uzunlukları eşitle
min_len = min(len(y_test), len(y_pred))
y_test = y_test[:min_len]
y_pred = y_pred[:min_len]

# 📊 Metrikler
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 🖼 Grafik çiz
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Gerçek Değerler", linewidth=2)
plt.plot(y_pred, label="Tahmin Edilen Değerler", linewidth=2)
plt.title(f"Nihai PV Güç Tahmini\nMSE: {mse:.2f} | R²: {r2:.4f}")
plt.xlabel("Zaman")
plt.ylabel("Güç (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()