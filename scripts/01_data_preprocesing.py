import pandas as pd
from pathlib import Path

# =========================
# CONFIGURACIÓN DE RUTAS
# =========================

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Archivos de entrada y salida
file_paths = [
    ("datos201912_origen.csv", "datos_calidad_transformados_2019.csv"),
    ("datos202012_origen.csv", "datos_calidad_transformados_2020.csv"),
    ("datos202112_origen.csv", "datos_calidad_transformados_2021.csv"),
    ("datos202212_origen.csv", "datos_calidad_transformados_2022.csv"),
    ("datos202312_origen.csv", "datos_calidad_transformados_2023.csv"),
    ("datos202406_origen.csv", "datos_calidad_transformados_2024.csv"),
]

# =========================
# CARGA DE DATOS AUXILIARES
# =========================

df_magnitudes = pd.read_csv(DATA_DIR / "calidad_magnitudes.csv", encoding="utf-8-sig")
df_estaciones = pd.read_csv(DATA_DIR / "calidad_estaciones.csv", encoding="utf-8-sig")

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def transform_data(df_origen):
    
    df_origen = df_origen.rename(columns={
        "ESTACION": "Cod_corto",
        "MAGNITUD": "Codigo",
        "ANO": "year",
        "MES": "month"
    })

    # reshape
    df_melted = df_origen.melt(
        id_vars=["Cod_corto", "Codigo", "year", "month"],
        var_name="DIA",
        value_name="VALOR"
    )

    df_melted["DIA"] = df_melted["DIA"].str.extract(r"(\d+)").astype(int)

    df_melted["fecha"] = pd.to_datetime(
        df_melted[["year", "month", "DIA"]].astype(str).agg("-".join, axis=1),
        errors="coerce"
    )

    df_melted = df_melted.dropna(subset=["fecha"])

    # filtrar magnitudes
    relevant_magnitudes = [1, 6, 7, 8, 9, 10, 12]
    df_filtered = df_melted[df_melted["Codigo"].isin(relevant_magnitudes)]

    # joins
    df_combined = (
        df_filtered
        .merge(df_estaciones, on="Cod_corto")
        .merge(df_magnitudes, on="Codigo")
    )

    df_final = df_combined[
        ["fecha", "Cod_corto", "Barrio", "Distrito", "Abreviatura", "VALOR"]
    ]

    # pivot
    df_pivot = df_final.pivot_table(
        index=["fecha", "Cod_corto", "Barrio", "Distrito"],
        columns="Abreviatura",
        values="VALOR"
    ).reset_index()

    # columnas finales
    valid_columns = [
        "fecha", "Cod_corto", "Barrio", "Distrito",
        "SO2", "CO", "NO", "NO2", "PM2.5", "PM10", "NOx"
    ]

    df_pivot = df_pivot[valid_columns]

    return df_pivot

# =========================
# EJECUCIÓN
# =========================

def main():

    generated_files = []

    for input_name, output_name in file_paths:

        input_path = DATA_DIR / input_name
        output_path = OUTPUT_DIR / output_name

        df_origen = pd.read_csv(input_path, encoding="latin1", sep=";")
        df_transformed = transform_data(df_origen)

        df_transformed.to_csv(output_path, index=False)

        generated_files.append(output_path)

    print("Generated files:")
    for f in generated_files:
        print(f)

if __name__ == "__main__":
    main()