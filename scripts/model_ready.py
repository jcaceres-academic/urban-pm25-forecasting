import pandas as pd

# Cargar dataset limpio
df = pd.read_csv("pm25_lisbon_2021_2024_clean.csv")

# Convertir fecha
df["date"] = pd.to_datetime(df["date"])

# Codificar station_type
df = pd.get_dummies(df, columns=["station_type"], drop_first=True)

# Seleccionar columnas finales
df_model = df[[
    "date",
    "pm25_daily",
    "lag1",
    "lag2",
    "ma3",
    "doy",
    "station_type_Industrial",
    "station_type_Traffic",
    "class"
]]

# Guardar dataset listo para modelado
df_model.to_csv("pm25_lisbon_2021_2024_model_ready.csv", index=False)

print("Dataset model_ready creado:", df_model.shape)
