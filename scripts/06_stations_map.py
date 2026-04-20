import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path

DATA_PATH = Path("data/raw/estaciones_control_aire.csv")
SHAPE_PATH = Path("data/shapes/Distritos.shp")
OUTPUT_DIR = Path("images")

OUTPUT_DIR.mkdir(exist_ok=True)

def main():

    districts = gpd.read_file(SHAPE_PATH)

    df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LONGITUD"], df["LATITUD"]),
        crs="EPSG:4326"
    )

    # reproyección
    districts = districts.to_crs(epsg=3857)
    gdf_points = gdf_points.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 10))

    xmin, ymin, xmax, ymax = gdf_points.total_bounds
    margin = 2000

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    img, ext = ctx.bounds2img(
        xmin - margin,
        ymin - margin,
        xmax + margin,
        ymax + margin,
        source=ctx.providers.OpenStreetMap.Mapnik
    )

    ax.imshow(img, extent=ext, alpha=0.5, zorder=1)

    districts.plot(
        ax=ax,
        color="none",
        edgecolor=(0, 0, 0, 0.4),
        linewidth=0.6,
        zorder=2
    )

    gdf_points.plot(
        ax=ax,
        color="#f28e2b",
        markersize=80,
        edgecolor="black",
        linewidth=0.5,
        zorder=3
    )

    ax.set_title("PM2.5 Monitoring Stations in Madrid")
    ax.set_axis_off()

    plt.savefig(
        OUTPUT_DIR / "Fig_PM25_Stations_Madrid.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

if __name__ == "__main__":
    main()