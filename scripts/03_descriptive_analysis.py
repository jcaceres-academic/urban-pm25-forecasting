import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("images")

OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    dfs = []
    for year in range(2019, 2025):
        dfs.append(pd.read_csv(DATA_DIR / f"datos_calidad_transformados_{year}.csv"))
    return pd.concat(dfs)

def main():

    df = load_data()

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df[(df["PM2.5"] >= 0) & (df["PM2.5"] <= 100)]

    df["year"] = df["fecha"].dt.year

    stats = df.groupby("year")["PM2.5"].agg(
        ["mean", "median", "std", "min", "max"]
    ).reset_index()

    # plot líneas
    plt.figure(figsize=(12, 6))
    for col in ["mean", "median", "std"]:
        plt.plot(stats["year"], stats[col], label=col)

    plt.legend()
    plt.title("PM2.5 descriptive statistics (2019–2024)")
    plt.savefig(OUTPUT_DIR / "Fig_Descriptive_PM25.png", dpi=300)
    plt.close()

    # violin
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="year", y="PM2.5", data=df)
    plt.title("Distribution of PM2.5 by year")
    plt.savefig(OUTPUT_DIR / "Fig_Violin_PM25.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()