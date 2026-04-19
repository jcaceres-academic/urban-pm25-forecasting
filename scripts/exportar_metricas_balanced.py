import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier

# ===============================
# 1. CARGAR DATASET LIMPIO
# ===============================

df = pd.read_csv("pm25_lisbon_2021_2024_clean.csv")
df["date"] = pd.to_datetime(df["date"])

# ===============================
# 2. SPLIT TEMPORAL
# ===============================

train = df[df["date"] < "2023-01-01"]
test  = df[df["date"] >= "2023-01-01"]

features = ["pm25_daily", "lag1", "lag2", "ma3", "doy"]

X_train = train[features]
X_test  = test[features]

y_train = train["class"]
y_test  = test["class"]

# ===============================
# 3. CODIFICACIÓN
# ===============================

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ===============================
# 4. MODELOS
# ===============================

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(eval_metric="mlogloss"),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500)
}

results = []

for name, model in models.items():

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_inf = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_inf

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "MacroF1": f1_score(y_test, y_pred, average="macro"),
        "BalancedAcc": balanced_accuracy_score(y_test, y_pred),
        "TrainTime": train_time,
        "InferenceTime": inference_time
    })

metrics = pd.DataFrame(results)

# ===============================
# 5. GUARDAR MÉTRICAS COMPLETAS
# ===============================

metrics.to_csv("model_results_balanced_full.csv", index=False)
print("Archivo 'model_results_balanced_full.csv' guardado.")

# ===============================
# 6. RADAR
# ===============================

radar = metrics.copy()

# Invertir tiempos (menos es mejor)
radar["TrainTime"] = 1 / radar["TrainTime"]
radar["InferenceTime"] = 1 / radar["InferenceTime"]

cols = ["Accuracy", "MacroF1", "BalancedAcc",
        "TrainTime", "InferenceTime"]

# Normalización 0-1
for c in cols:
    radar[c] = (radar[c] - radar[c].min()) / (radar[c].max() - radar[c].min())

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

print("Radar guardado como 'radar_balanced_models.png'")
