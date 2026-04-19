import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =====================================
# CARGAR RESULTADOS BALANCED SCENARIO
# =====================================

df = pd.read_csv("model_results_balanced_full.csv")

print(df)

# =====================================
# NORMALIZACIÓN
# =====================================

scaler = MinMaxScaler()

# Métricas de rendimiento
performance_cols = ["Accuracy", "MacroF1", "BalancedAcc"]
df_perf = pd.DataFrame(
    scaler.fit_transform(df[performance_cols]),
    columns=performance_cols
)

df["PerformanceScore"] = df_perf.mean(axis=1)

# Métricas de eficiencia (invertimos tiempos)
eff_cols = ["TrainTime", "InferenceTime"]

df_eff = pd.DataFrame(
    scaler.fit_transform(df[eff_cols]),
    columns=eff_cols
)

# Invertir (menos tiempo = mejor)
df_eff = 1 - df_eff

df["EfficiencyScore"] = df_eff.mean(axis=1)

print("\nScores agregados:")
print(df[["Model", "PerformanceScore", "EfficiencyScore"]])

# =====================================
# GRÁFICO TRADE-OFF
# =====================================

plt.figure(figsize=(9,7))

# Colores personalizados
colors = {
    "LogReg": "#1f77b4",
    "RandomForest": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "MLP": "#d62728"
}

for _, row in df.iterrows():
    plt.scatter(
        row["PerformanceScore"],
        row["EfficiencyScore"],
        s=500 * row["MacroF1"],   # tamaño proporcional al MacroF1
        color=colors[row["Model"]],
        alpha=0.85,
        edgecolor='black'
    )
    plt.text(
        row["PerformanceScore"] + 0.015,
        row["EfficiencyScore"] + 0.015,
        row["Model"],
        fontsize=11
    )

# Frontera de Pareto (simple)
df_sorted = df.sort_values("PerformanceScore")
plt.plot(
    df_sorted["PerformanceScore"],
    df_sorted["EfficiencyScore"],
    linestyle="--",
    color="grey",
    alpha=0.6
)

plt.xlabel("Performance Score", fontsize=12)
plt.ylabel("Efficiency Score", fontsize=12)
plt.title("Performance–Efficiency Trade-off Map\nLisbon PM2.5 (2021–2023)", fontsize=14)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("tradeoff_map_lisbon_balanced_v2.png", dpi=300)
plt.show()
