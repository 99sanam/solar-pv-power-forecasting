
import numpy as np
import os

# 📁 Tahmin klasörü
prediction_folder = "predictions"

# 🌀 IMF_1 ... IMF_10 ve Residual tahminlerini yükle
y_preds = []

for i in range(1, 11):
    file_path = os.path.join(prediction_folder, f"y_pred_{i}.npy")
    if os.path.exists(file_path):
        y_pred = np.load(file_path)
        y_preds.append(y_pred)
    else:
        print(f"⚠ {file_path} bulunamadı, atlanıyor.")

# Residual bileşenini de ekle
residual_path = os.path.join(prediction_folder, "y_pred_residual.npy")
if os.path.exists(residual_path):
    y_residual = np.load(residual_path)
    y_preds.append(y_residual)
else:
    print("⚠ Residual tahmin dosyası bulunamadı!")

# ✅ Tüm tahminleri topla (satır satır)
if y_preds:
    final_prediction = np.sum(y_preds, axis=0)
    np.save(os.path.join(prediction_folder, "y_pred_final.npy"), final_prediction)
    print("\n✅ Nihai PV tahmini başarıyla oluşturuldu ve 'y_pred_final.npy' olarak kaydedildi.")
    print(f"Tahmin uzunluğu: {final_prediction.shape[0]}")
else:
    print("❌ Hiçbir tahmin dosyası yüklenemedi!")