# ============================================
# 08_district_forecasting.py
# District-level Prophet–LSTM Forecasting
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from unidecode import unidecode

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

data = data[(data["PM2.5"] >= 0) & (data["PM2.5"] <= 100)]

data["Distrito"] = data["Distrito"].apply(
    lambda x: unidecode(x).upper().replace(" ", "").replace("-", "")
)

districts = [
    "ARGANZUELA", "SALAMANCA", "MONCLOAARAVACA",
    "HORTALEZA", "CARABANCHEL", "CHAMBERI", "CHAMARTIN"
]

for distrito in districts:

    df = data[data["Distrito"] == distrito]

    if df.empty:
        continue

    daily = df.groupby("fecha")["PM2.5"].mean().reset_index()

    prophet_df = daily.rename(columns={"fecha": "ds", "PM2.5": "y"})

    model_p = Prophet(daily_seasonality=True)
    model_p.fit(prophet_df)

    future = model_p.make_future_dataframe(periods=180)
    forecast = model_p.predict(future)

    pred = forecast[["ds", "yhat"]].rename(columns={"ds": "fecha"})

    combined = pd.merge(daily, pred, on="fecha", how="left")
    combined["residuals"] = combined["PM2.5"] - combined["yhat"]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(combined["residuals"].dropna().values.reshape(-1, 1))

    def create_dataset(dataset, step=10):
        X, y = [], []
        for i in range(len(dataset) - step - 1):
            X.append(dataset[i:(i + step), 0])
            y.append(dataset[i + step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled, 10)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)

    print(f"✔ Forecast completed for {distrito}")