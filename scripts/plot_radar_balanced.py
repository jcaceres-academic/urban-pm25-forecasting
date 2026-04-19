import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CARGAR RESULTADOS CLÁSICOS
# ===============================

df = pd.read_csv("model_ranking_lisbon_2021_2023_balanced.csv")

# Necesitamos también las métricas clásicas
metrics = pd.read_csv("model_results_balanced_full.csv")  
# (Si no lo guardaste, te doy luego el código para exportarlo)

# ===============================
# NORMALIZACIÓN
# ===============================

cols = ["Accuracy", "MacroF1", "BalancedAcc",
        "TrainTime", "InferenceTime"]

radar = metrics.copy()

# Invertimos tiempos (menos es mejor)
radar["TrainTime"] = 1 / radar["TrainTime"]
radar["InferenceTime"] = 1 / radar["InferenceTime"]

# Normalización 0-1
for c in cols:
    radar[c] = (radar[c] - radar[c].min()) / (radar[c].max() - radar[c].min())

# ===============================
# RADAR
# ===============================

categories = cols
N = len(categories)

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for i, row in radar.iterrows():
    values = row[categories].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=row["Model"])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

plt.title("Model Profiles – Balanced Scenario")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig("radar_balanced_models.png", dpi=300)
plt.show()
