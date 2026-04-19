import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Force clean white style (important for export)
# --------------------------------------------------

plt.style.use('default')
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

# --------------------------------------------------
# Raw performance data
# --------------------------------------------------

data = {
    "Model": ["LogReg", "RandomForest", "XGBoost", "MLP"],
    "MacroF1": [0.861, 0.779, 0.838, 0.962],
    "BalancedAcc": [0.799, 0.738, 0.820, 0.943],
    "TrainTime": [0.02, 0.45, 0.13, 1.46],
    "InferenceTime": [0.0002, 0.016, 0.002, 0.001]
}

df = pd.DataFrame(data)

# --------------------------------------------------
# Composite metrics
# --------------------------------------------------

df["Performance"] = (df["MacroF1"] + df["BalancedAcc"]) / 2
df["TotalTime"] = df["TrainTime"] + df["InferenceTime"]

df["Performance_norm"] = (
    df["Performance"] - df["Performance"].min()
) / (df["Performance"].max() - df["Performance"].min())

df["LogCost"] = np.log10(df["TotalTime"])

# --------------------------------------------------
# Create figure
# --------------------------------------------------

plt.figure(figsize=(7,6))
ax = plt.gca()

# Remove top/right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

base_color = "#1F3A5F"
highlight_color = "#FAB004"

# --------------------------------------------------
# Scatter points
# --------------------------------------------------

for i in range(len(df)):
    x = df["Performance_norm"][i]
    y = df["LogCost"][i]
    model = df["Model"][i]
    
    if model == "MLP":
        # Halo
        plt.scatter(x, y,
                    s=900,
                    color=highlight_color,
                    alpha=0.18,
                    edgecolors='none',
                    zorder=1)
        
        # Main point
        plt.scatter(x, y,
                    s=400,
                    color=highlight_color,
                    edgecolor='black',
                    linewidth=0.8,
                    zorder=2)
    else:
        plt.scatter(x, y,
                    s=400,
                    color=base_color,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=2)

# --------------------------------------------------
# Efficient frontier (excluding dominated model)
# --------------------------------------------------

df_frontier = df[df["Model"] != "RandomForest"]
df_frontier = df_frontier.sort_values("LogCost")

plt.plot(df_frontier["Performance_norm"],
         df_frontier["LogCost"],
         linestyle="--",
         color="#888888",
         linewidth=0.8,
         zorder=0)

# -----------------------------
# Labels
# -----------------------------

for i in range(len(df)):
    x = df["Performance_norm"][i]
    y = df["LogCost"][i]
    model = df["Model"][i]
    
    if model == "RandomForest":
        plt.text(x + 0.05, y,
                 model,
                 ha='left',
                 va='center',
                 fontsize=10)
        
    elif model == "MLP":
        plt.text(x, y - 0.12,
                 model,
                 ha='center',
                 va='top',
                 fontsize=10)
        
    elif model == "LogReg":
        plt.text(x + 0.10, y,
                 model,
                 ha='center',
                 va='top',
                 fontsize=10)
        
    else:  # XGBoost
        plt.text(x, y + 0.10,
                 model,
                 ha='center',
                 va='bottom',
                 fontsize=10)

# --------------------------------------------------
# Axes formatting
# --------------------------------------------------

plt.xlabel("Normalized Performance Score")
plt.ylabel("Log10 Total Computational Time (seconds)")

plt.grid(alpha=0.25, linestyle="--", linewidth=0.7)

plt.tight_layout()
plt.savefig("Figure_Performance_Cost_Map_300dpi.png", dpi=300)
plt.show()