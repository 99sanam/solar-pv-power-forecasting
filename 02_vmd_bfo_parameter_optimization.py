
import pandas as pd
import numpy as np
from scipy.signal import hilbert
import random
from vmdpy import VMD  # pip install vmdpy

# 🟣 1. Veri Yükleme (Yeni veri seti)
df = pd.read_csv("cleaned_dataset.csv")
y = df['Active_Power'].values

# 🟣 2. Eğitim verisini ayır (%70)
train_size = int(len(y) * 0.7)
y_train = y[:train_size]

# 🔵 3. BFO Parametreleri
n_fish = 25
max_iter = 50
K_min, K_max = 2, 10
alpha_min, alpha_max = 200, 1000
SD0 = 1
p = 1

# 🔵 4. Envelope Entropy (Makale Eq. 18-20)
def envelope_entropy(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    total = np.sum(amplitude_envelope)
    if total == 0:
        return 0
    p_i = amplitude_envelope / total
    entropy = -np.sum(p_i * np.log(p_i + 1e-10))
    return entropy

# 🔵 5. Cost Function: VMD + Entropy
def cost_function(K, alpha):
    try:
        u, _, _ = VMD(
            y_train,
            alpha=alpha,
            tau=0,
            K=K,
            DC=0,
            init=1,
            tol=1e-7
        )
        reconstructed = np.sum(u, axis=0)
        return envelope_entropy(reconstructed)
    except:
        return np.inf

# 🔵 6. Başlangıç Popülasyonu
population = []
for _ in range(n_fish):
    K = random.randint(K_min, K_max)
    alpha = random.uniform(alpha_min, alpha_max)
    cost = cost_function(K, alpha)
    population.append({'K': K, 'alpha': alpha, 'cost': cost})

# 🔵 7. En iyi bireyi bul
best = min(population, key=lambda x: x['cost'])

# 🔵 8. BFO Döngüsü
for it in range(max_iter):
    P = abs(1 - it / (np.sqrt(1 + it ** 2))) + random.random() / (it + 1e-5)**0.5
    U = (1 - it / max_iter) * np.cos(it)
    J = 1 * U
    SD = SD0 - (p / np.pi) * np.arctan(it * p / np.pi)

    new_population = []
    for fish in population:
        if random.random() < P:
            new_K = fish['K'] + random.random() * (best['K'] - J * fish['K'])
            new_alpha = fish['alpha'] + random.random() * (best['alpha'] - J * fish['alpha'])
        else:
            new_K = random.randint(K_min, K_max)
            new_alpha = random.uniform(alpha_min, alpha_max)

        new_K = int(np.clip(round(new_K), K_min, K_max))
        new_alpha = np.clip(new_alpha, alpha_min, alpha_max)
        cost = cost_function(new_K, new_alpha)

        new_population.append({'K': new_K, 'alpha': new_alpha, 'cost': cost})

    population = new_population
    best = min(population, key=lambda x: x['cost'])

    print(f"Iter {it+1}/{max_iter} → En iyi entropy: {best['cost']:.5f}")

# 🟢 9. Sonuçları Yazdır
print("\n🎯 En iyi VMD parametreleri (BFO ile):")
print(f"K (mod sayısı): {best['K']}")
print(f"Alpha         : {best['alpha']:.2f}")
print(f"Minimum Envelope Entropy: {best['cost']:.5f}")
