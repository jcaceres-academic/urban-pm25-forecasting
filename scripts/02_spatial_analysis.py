import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from unidecode import unidecode

# =========================
# RUTAS
# =========================

DATA_DIR = Path("data/processed")
SHAPE_PATH = Path("data/shapes/Distritos.shp")
OUTPUT_DIR = Path("images")

OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# FUNCIONES AUXILIARES
# =========================

def normalize(text):
    return unidecode(text).upper().replace(" ", "").replace("-", "")

# =========================
# CARGA
# =========================

def load_data():
    dfs = []
    for year in range(2019, 2025):
        file = DATA_DIR / f"datos_calidad_transformados_{year}.csv"
        dfs.append(pd.read_csv(file))
    return pd.concat(dfs)

# =========================
# MAIN
# =========================

def main():

    data = load_data()

    data["fecha"] = pd.to_datetime(data["fecha"])
    data["year"] = data["fecha"].dt.year

    districts = gpd.read_file(SHAPE_PATH)

    data["Distrito"] = data["Distrito"].apply(normalize)
    districts["NOMBRE"] = districts["NOMBRE"].apply(normalize)

    for year in range(2019, 2025):

        year_data = data[data["year"] == year]

        district_pm25 = (
            year_data.groupby("Distrito")["PM2.5"]
            .mean()
            .reset_index()
        )

        merged = districts.merge(
            district_pm25,
            left_on="NOMBRE",
            right_on="Distrito",
            how="left"
        )

        fig, ax = plt.subplots(figsize=(10, 10))

        merged.plot(
            column="PM2.5",
            cmap="RdYlGn_r",
            linewidth=0.6,
            edgecolor="black",
            legend=True,
            ax=ax
        )

        ax.set_title(f"PM2.5 levels by district ({year})")
        ax.set_axis_off()

        plt.savefig(
            OUTPUT_DIR / f"PM25_LevelsDistrict_{year}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

if __name__ == "__main__":
    main()