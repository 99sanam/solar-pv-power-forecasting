
import numpy as np
import os

# 1. Giriş (X_train) verisini yükle
X_train = np.load("X_train.npy")

# 2. LSTM veri setleri için klasör oluştur
os.makedirs("lstm_datasets", exist_ok=True)

# 3. IMF sayısı (BFO sonucu K=10)
K = 10

# 4. IMF dosyalarını sırayla işle
for i in range(1, K + 1):
    y_target_path = f"vmd_output/IMF_{i}.npy"
    if os.path.exists(y_target_path):
        y_target = np.load(y_target_path)
        min_len = min(len(y_target), len(X_train))  # Güvenli kırpma
        combined = np.hstack((X_train[:min_len], y_target[:min_len].reshape(-1, 1)))
        np.save(f"lstm_datasets/IMF_{i}_XY.npy", combined)
        print(f"✔ IMF_{i}_XY.npy kaydedildi.")
    else:
        print(f"⚠ IMF_{i}.npy dosyası bulunamadı.")

# 5. Residual dosyası
residual_path = "vmd_output/Residual.npy"
if os.path.exists(residual_path):
    residual = np.load(residual_path)
    min_len = min(len(residual), len(X_train))  # Güvenli kırpma
    combined_res = np.hstack((X_train[:min_len], residual[:min_len].reshape(-1, 1)))
    np.save("lstm_datasets/Residual_XY.npy", combined_res)
    print(f"✔ Residual_XY.npy kaydedildi.")
else:
    print("⚠ Residual.npy dosyası bulunamadı.")

print(f"\n✅ {K} IMF bileşeni ve 1 residual dosyası başarıyla 'lstm_datasets' klasörüne kaydedildi.")
