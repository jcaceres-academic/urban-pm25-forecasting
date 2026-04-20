import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from unidecode import unidecode

SHAPE_PATH = Path("data/shapes/Distritos.shp")
OUTPUT_DIR = Path("images")

OUTPUT_DIR.mkdir(exist_ok=True)

PM25_DISTRICTS = [
    "CHAMARTIN",
    "CHAMBERI",
    "SALAMANCA",
    "ARGANZUELA",
    "CARABANCHEL",
    "HORTALEZA",
    "MONCLOA-ARAVACA"
]

def normalize(text):
    return unidecode(text).upper().replace(" ", "").replace("-", "")

def main():

    gdf = gpd.read_file(SHAPE_PATH)

    gdf["NOMBRE"] = gdf["NOMBRE"].apply(normalize)

    gdf["PM25"] = gdf["NOMBRE"].apply(
        lambda x: "Yes" if x in PM25_DISTRICTS else "No"
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    gdf[gdf["PM25"] == "Yes"].plot(
        ax=ax,
        color="skyblue",
        edgecolor="black"
    )

    gdf[gdf["PM25"] == "No"].plot(
        ax=ax,
        color="lightgrey",
        edgecolor="black"
    )

    ax.set_title("Districts measuring PM2.5")
    ax.set_axis_off()

    plt.savefig(
        OUTPUT_DIR / "Fig_Districts_PM25.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

if __name__ == "__main__":
    main()