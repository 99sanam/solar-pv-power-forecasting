# Solar PV Power Forecasting ⚡️

A machine learning project for **photovoltaic (PV) power forecasting**.  
The pipeline follows: **Z-Score → VMD → BFO (optimizer) → LSTM (per-IMF) → Final Prediction**.

## 🚀 Features
- Data preprocessing (Z-Score normalization, filtering 07:00–19:00)
- VMD decomposition (IMFs + residual)
- BFO optimization (VMD parameters and LSTM hyperparameters)
- LSTM models trained per IMF
- Metrics: MSE, MAE, MAPE, R²
- Result plots (y_true vs y_pred)

## 📂 Project Structure
## ▶️ Quick Start
```bash
pip install -r requirements.txt
src/ # Python scripts (preprocessing, vmd, bfo, lstm, evaluation)
data/ # cleaned sample dataset
requirements.txt


👤 Author

Sanam Faizi
Software Engineering Student, Istanbul Topkapı University
