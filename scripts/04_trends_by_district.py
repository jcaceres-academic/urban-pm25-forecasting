import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("images")

OUTPUT_DIR.mkdir(exist_ok=True)

VALID_DISTRICTS = [
    "SALAMANCA",
    "MONCLOA-ARAVACA",
    "CHAMBERI",
    "ARGANZUELA",
    "CHAMARTIN",
    "CARABANCHEL",
    "HORTALEZA"
]

def load_data():
    dfs = []
    for year in range(2019, 2024):
        dfs.append(pd.read_csv(DATA_DIR / f"datos_calidad_transformados_{year}.csv"))
    return pd.concat(dfs)

def main():

    df = load_data()

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df[(df["PM2.5"] >= 0) & (df["PM2.5"] <= 100)]

    df["year"] = df["fecha"].dt.year

    df = df[df["Distrito"].isin(VALID_DISTRICTS)]

    plt.figure(figsize=(14, 8))

    for district in VALID_DISTRICTS:
        subset = df[df["Distrito"] == district]
        if not subset.empty:
            sns.lineplot(
                x="year",
                y="PM2.5",
                data=subset,
                label=district,
                ci=None
            )

    plt.title("PM2.5 trends by district (2019–2023)")
    plt.xlabel("Year")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "Fig_Trends_PM25_District.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

if __name__ == "__main__":
    main()