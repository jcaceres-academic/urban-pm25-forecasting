# ============================================
# 07_hybrid_forecasting.py
# Prophet–LSTM Hybrid Model for PM2.5 Forecasting
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===============================
# 1. LOAD DATA
# ===============================

DATA_PATH = "../data/"

files = [
    "datos_calidad_transformados_2019.csv",
    "datos_calidad_transformados_2020.csv",
    "datos_calidad_transformados_2021.csv",
    "datos_calidad_transformados_2022.csv",
    "datos_calidad_transformados_2023.csv",
    "datos_calidad_transformados_2024.csv",
]

dfs = [pd.read_csv(DATA_PATH + f) for f in files]
data = pd.concat(dfs)

data["fecha"] = pd.to_datetime(data["fecha"])

# Clean data
data = data[(data["PM2.5"] >= 0) & (data["PM2.5"] <= 100)]

# Daily aggregation
daily_pm25 = data.groupby("fecha")["PM2.5"].mean().reset_index()

# ===============================
# 2. PROPHET MODEL
# ===============================

prophet_df = daily_pm25.rename(columns={"fecha": "ds", "PM2.5": "y"})

model_prophet = Prophet()
model_prophet.fit(prophet_df)

future = model_prophet.make_future_dataframe(periods=180)
forecast = model_prophet.predict(future)

# ===============================
# 3. RESIDUALS
# ===============================

pred_prophet = forecast[["ds", "yhat"]].rename(columns={"ds": "fecha"})

combined = pd.merge(daily_pm25, pred_prophet, on="fecha", how="left")
combined["residuals"] = combined["PM2.5"] - combined["yhat"]

# ===============================
# 4. NORMALIZATION
# ===============================

scaler = MinMaxScaler()
residuals = combined["residuals"].dropna().values.reshape(-1, 1)
scaled_residuals = scaler.fit_transform(residuals)

# ===============================
# 5. LSTM DATASET
# ===============================

def create_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X_train, y_train = create_dataset(scaled_residuals, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# ===============================
# 6. LSTM MODEL
# ===============================

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

# ===============================
# 7. FUTURE RESIDUAL PREDICTION
# ===============================

temp_input = list(scaled_residuals[-time_step:].flatten())
lst_output = []

for _ in range(180):
    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(yhat[0][0])
    lst_output.append(yhat[0])

predicted_residuals = scaler.inverse_transform(lst_output)

# ===============================
# 8. FINAL HYBRID FORECAST
# ===============================

future_dates = future["ds"][-180:].reset_index(drop=True)

final_df = pd.DataFrame({
    "fecha": future_dates,
    "prophet": forecast["yhat"][-180:].values,
    "prophet_lower": forecast["yhat_lower"][-180:].values,
    "prophet_upper": forecast["yhat_upper"][-180:].values,
    "residuals": predicted_residuals.flatten()
})

final_df["PM2.5"] = final_df["prophet"] + final_df["residuals"]

# Confidence interval adjustment
scale = 0.6
width = final_df["prophet_upper"] - final_df["prophet"]

final_df["lower"] = final_df["PM2.5"] - scale * width
final_df["upper"] = final_df["PM2.5"] + scale * width

# ===============================
# 9. PLOT
# ===============================

plt.figure(figsize=(14, 8))

plt.plot(daily_pm25["fecha"], daily_pm25["PM2.5"], label="Historical", color="blue")

real_2024 = daily_pm25[daily_pm25["fecha"] >= "2024-01-01"]
plt.plot(real_2024["fecha"], real_2024["PM2.5"], label="Observed 2024", color="red")

plt.plot(final_df["fecha"], final_df["PM2.5"], label="Forecast", color="green")

plt.fill_between(final_df["fecha"], final_df["lower"], final_df["upper"],
                 color="green", alpha=0.25)

plt.axvline(pd.Timestamp("2024-07-01"), linestyle="--", color="red")

plt.title("PM2.5 Forecast - Prophet–LSTM")
plt.xlabel("Year")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.grid(True)

plt.savefig("../images/Fig_Forecast_Madrid_Prophet_LSTM.png", dpi=300)
plt.show()