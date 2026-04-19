import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# =====================================================
# 1. CARGA DATASET
# =====================================================

df = pd.read_csv("pm25_lisbon_2021_2024_model_ready.csv")
df["date"] = pd.to_datetime(df["date"])

print("Min date:", df["date"].min())
print("Max date:", df["date"].max())
print("\nObservaciones por año:")
print(df.groupby(df["date"].dt.year).size())

# =====================================================
# 2. SPLIT TEMPORAL
# =====================================================

train = df[df["date"].dt.year <= 2022]
test  = df[df["date"].dt.year == 2023]

print("\nTrain:", train.shape)
print("Test:", test.shape)

# =====================================================
# 3. FEATURES CORRECTAS (SIN pm25_daily)
# =====================================================

features = [
    "lag1",
    "lag2",
    "ma3",
    "doy",
    "station_type_Industrial",
    "station_type_Traffic"
]

X_train = train[features]
X_test  = test[features]

# =====================================================
# 4. CODIFICACIÓN DE CLASES
# =====================================================

le = LabelEncoder()
y_train = le.fit_transform(train["class"])
y_test  = le.transform(test["class"])

print("\nClases codificadas:", le.classes_)

# =====================================================
# 5. ESCALADO
# =====================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =====================================================
# 6. DEFINICIÓN DE MODELOS
# =====================================================

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss'),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500)
}

results = []

# =====================================================
# 7. ENTRENAMIENTO Y MÉTRICAS
# =====================================================

for name, model in models.items():

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(X_test)
    inference_time = time.time() - start

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    bal = balanced_accuracy_score(y_test, preds)

    results.append([
        name, acc, f1, bal, train_time, inference_time
    ])

results_df = pd.DataFrame(results, columns=[
    "Model","Accuracy","MacroF1","BalancedAcc",
    "TrainTime","InferenceTime"
])

print("\nResultados clásicos:")
print(results_df)

# =====================================================
# 8. AÑADIR COMPLEJIDAD (ORDINAL)
# =====================================================

complexity_map = {
    "LogReg": 1,
    "RandomForest": 3,
    "XGBoost": 4,
    "MLP": 5
}

results_df["Complexity"] = results_df["Model"].map(complexity_map)

# =====================================================
# 9. NORMALIZACIÓN MIN-MAX
# =====================================================

criteria = [
    "Accuracy",
    "MacroF1",
    "BalancedAcc",
    "TrainTime",
    "InferenceTime",
    "Complexity"
]

df_mc = results_df.copy()

for c in criteria:
    if c in ["TrainTime","InferenceTime","Complexity"]:
        df_mc[c] = (df_mc[c].max() - df_mc[c]) / (df_mc[c].max() - df_mc[c].min())
    else:
        df_mc[c] = (df_mc[c] - df_mc[c].min()) / (df_mc[c].max() - df_mc[c].min())

# =====================================================
# 10. PESOS MULTICRITERIO
# =====================================================

weights = np.array([
    0.20,  # Accuracy
    0.20,  # MacroF1
    0.20,  # BalancedAcc
    0.15,  # TrainTime
    0.10,  # InferenceTime
    0.15   # Complexity
])


df_mc["MC_Score"] = df_mc[criteria].values.dot(weights)

# =====================================================
# 11. RANKING FINAL
# =====================================================

ranking = df_mc[["Model","MC_Score"]].sort_values(
    "MC_Score", ascending=False
)

print("\nRanking multicriterio:")
print(ranking)

# =====================================================
# 12. GUARDAR RESULTADOS
# =====================================================

ranking.to_csv("model_ranking_lisbon_2021_2023_balanced.csv", index=False)

print("\nArchivo 'model_ranking_lisbon_2021_2023_balanced.csv' guardado.")
