import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Temizlenmiş CSV dosyasını oku
df = pd.read_csv("cleaned_dataset.csv")

# 2. Giriş (X) ve hedef (y) değişkenlerini ayır
X = df[['Weather_Temperature_Celsius',
        'Relative_Humidity',
        'Global_Horizontal_Radiation',
        'Diffuse_Horizontal_Radiation']].values

y = df['Active_Power'].values

# 3. Z-Score normalizasyonu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. %70 eğitim, %30 test böl
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, shuffle=False
)

# 5. Numpy olarak kaydet
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# 6. Görselleştir
plt.figure(figsize=(10, 5))
plt.boxplot(X_scaled)
plt.title("Z-Score Normalize Edilmiş Veriler (4 Özellik)")
plt.xticks([1, 2, 3, 4], ['Sıcaklık', 'Nem', 'GHI', 'DHI'])
plt.grid(True)
plt.show()
