import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. CARGAR RANKINGS
# =====================================================

perf = pd.read_csv("model_ranking_lisbon_2021_2023_performance.csv")
eff  = pd.read_csv("model_ranking_lisbon_2021_2023_efficiency.csv")
bal  = pd.read_csv("model_ranking_lisbon_2021_2023_balanced.csv")

perf["Scenario"] = "Performance"
eff["Scenario"]  = "Efficiency"
bal["Scenario"]  = "Balanced"

df = pd.concat([perf, eff, bal], ignore_index=True)

# =====================================================
# 2. REESTRUCTURAR PARA PLOT
# =====================================================

pivot = df.pivot(index="Scenario", columns="Model", values="MC_Score")

# Orden consistente
pivot = pivot.loc[["Performance", "Efficiency", "Balanced"]]

# =====================================================
# 3. PLOT
# =====================================================

plt.figure(figsize=(8,6))

for model in pivot.columns:
    plt.plot(
        pivot.index,
        pivot[model],
        marker="o",
        linewidth=2,
        label=model
    )

plt.title("Multi-Criteria Score Across Decision Scenarios")
plt.ylabel("MC Score")
plt.xlabel("Scenario")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ranking_scenarios_plot.png", dpi=300)
plt.show()

print("Figura guardada como ranking_scenarios_plot.png")
