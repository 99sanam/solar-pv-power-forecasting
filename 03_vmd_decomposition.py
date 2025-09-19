
import numpy as np
import os
from vmdpy import VMD  # pip install vmdpy

# 1. Eğitim verisini yükle
y = np.load("y_train.npy")  # Yeni veri setine göre oluşturulmuş olmalı

# 2. BFO ile elde edilen parametreler
K = 10
alpha = 202.79

# 3. Gerçek VMD uygulaması
u, u_hat, omega = VMD(
    y,
    alpha=alpha,    # regularization
    tau=0,          # noise-tolerance
    K=K,            # number of modes
    DC=0,           # no DC mode
    init=1,         # initialize frequencies uniformly
    tol=1e-7        # convergence tolerance
)

# 4. Uzunluk farkı sorununu düzelt (u ve y aynı uzunlukta olmalı)
min_len = min(len(y), u.shape[1])
y = y[:min_len]
u = u[:, :min_len]

# 5. Residual hesapla
residual = y - np.sum(u, axis=0)

# 6. Kayıt klasörü oluştur
os.makedirs("vmd_output", exist_ok=True)

# 7. IMF bileşenlerini kaydet
for i in range(u.shape[0]):
    np.save(f"vmd_output/IMF_{i+1}.npy", u[i])

# 8. Residual bileşenini kaydet
np.save("vmd_output/Residual.npy", residual)

print(f"✅ Gerçek VMD ile {K} IMF bileşeni ve residual başarıyla 'vmd_output/' klasörüne kaydedildi.")
