import pandas as pd
import numpy as np
import glob

# ===============================
# CONFIGURACIÓN
# ===============================

DATA_PATH = "../data/processed/"  # carpeta donde están los parquet
OUTPUT_FILE = "pm25_lisbon_2021_2024_clean.csv"

STATION_TYPE_MAP = {
    "PT03063": "Industrial",
    "PT03071": "Background",
    "PT03072": "Traffic",
    "PT03083": "Background"
}

START_DATE = "2021-01-01"
END_DATE   = "2024-12-31"

# ===============================
# 1. CARGA DE ARCHIVOS
# ===============================

files = glob.glob(DATA_PATH + "*.parquet")

dfs = []
for f in files:
    df = pd.read_parquet(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print("Registros iniciales:", data.shape)

# ===============================
# 2. FILTRADO TEMPORAL
# ===============================

data["Start"] = pd.to_datetime(data["Start"])

data = data[
    (data["Start"] >= START_DATE) &
    (data["Start"] <= END_DATE)
]

print("Tras filtrado temporal:", data.shape)

# ===============================
# 3. CONTROL DE CALIDAD
# ===============================

data = data[data["Validity"] == 1]
data = data[(data["Value"] >= 0) & (data["Value"] <= 500)]
data = data.dropna(subset=["Value"])

print("Tras control de calidad:", data.shape)

# ===============================
# 4. IDENTIFICACIÓN ESTACIÓN
# ===============================

data["station_code"] = data["Samplingpoint"].str.extract(r'SPO-(PT\d+)_')
data["station_type"] = data["station_code"].map(STATION_TYPE_MAP)

# ===============================
# 5. AGREGACIÓN DIARIA
# ===============================

data["date"] = data["Start"].dt.date

daily = (
    data.groupby(["station_code", "station_type", "date"])
    .agg(
        pm25_daily=("Value", "mean"),
        hours=("Value", "count")
    )
    .reset_index()
)

# Mínimo 18 horas (75%)
daily = daily[daily["hours"] >= 18]

print("Días tras agregación:", daily.shape)

# ===============================
# 6. ETIQUETADO DE CLASES
# ===============================

def classify_pm25(x):
    if x <= 10:
        return "Low"
    elif x <= 25:
        return "Moderate"
    else:
        return "High"

daily["class"] = daily["pm25_daily"].apply(classify_pm25)

# ===============================
# 7. FEATURES TEMPORALES
# ===============================

daily = daily.sort_values(["station_code", "date"])

daily["lag1"] = daily.groupby("station_code")["pm25_daily"].shift(1)
daily["lag2"] = daily.groupby("station_code")["pm25_daily"].shift(2)

daily["ma3"] = (
    daily.groupby("station_code")["pm25_daily"]
    .rolling(3)
    .mean()
    .reset_index(0, drop=True)
)

daily["doy"] = pd.to_datetime(daily["date"]).dt.dayofyear / 365

# ===============================
# 8. ELIMINAR NA POR LAGS
# ===============================

daily = daily.dropna()

print("Dataset final:", daily.shape)

# ===============================
# 8.1 DISTRIBUCIÓN TEST SET (2023)
# ===============================

daily["date"] = pd.to_datetime(daily["date"])

test_2023 = daily[daily["date"].dt.year == 2023]

n_test = len(test_2023)
n_high_test = (test_2023["class"] == "High").sum()

print("\n--- Test set 2023 summary ---")
print(f"Total observations (2023): {n_test}")
print(f"High pollution days (2023): {n_high_test}")
print(f"Percentage High (2023): {100 * n_high_test / n_test:.2f}%")

# ===============================
# 9. GUARDAR DATASET FINAL
# ===============================

daily.to_csv(OUTPUT_FILE, index=False)

print("Dataset guardado correctamente.")

df = pd.read_csv("pm25_lisbon_2021_2024_clean.csv")

print(df.groupby("station_code")["date"].nunique())
print(df["class"].value_counts())
print(df.groupby("station_type")["class"].value_counts())

print("Días por estación:")
print(daily.groupby("station_code")["date"].nunique())

print("\nMedia PM2.5 por estación:")
print(daily.groupby("station_code")["pm25_daily"].mean())

print("\nDesviación estándar por estación:")
print(daily.groupby("station_code")["pm25_daily"].std())

# Correlación entre Olivais y Laranjeiro
pivot = daily.pivot_table(
    index="date",
    columns="station_code",
    values="pm25_daily"
)

print(pivot["PT03071"].corr(pivot["PT03083"]))