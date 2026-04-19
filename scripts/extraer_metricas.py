import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# =====================================================
# 1. CARGA DATASET
# =====================================================

df = pd.read_csv("pm25_lisbon_2021_2024_model_ready.csv")
df["date"] = pd.to_datetime(df["date"])

# =====================================================
# 2. SPLIT TEMPORAL
# =====================================================

train = df[df["date"].dt.year <= 2022]
test  = df[df["date"].dt.year == 2023]

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
# 3. CODIFICACIÓN DE CLASES
# =====================================================

le = LabelEncoder()
y_train = le.fit_transform(train["class"])
y_test  = le.transform(test["class"])

# =====================================================
# 4. ESCALADO
# =====================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =====================================================
# 5. MODELOS
# =====================================================

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss'),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500)
}

high_metrics = []

# =====================================================
# 6. EXTRACCIÓN MÉTRICAS CLASE HIGH
# =====================================================

for name, model in models.items():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(
        y_test,
        preds,
        target_names=le.classes_,
        output_dict=True
    )

    high_precision = report["High"]["precision"]
    high_recall = report["High"]["recall"]
    high_f1 = report["High"]["f1-score"]

    high_metrics.append([
        name,
        high_precision,
        high_recall,
        high_f1
    ])

high_df = pd.DataFrame(high_metrics, columns=[
    "Model",
    "High_Precision",
    "High_Recall",
    "High_F1"
])

print("\nMétricas específicas clase HIGH:")
print(high_df.sort_values("High_F1", ascending=False))

# =====================================================
# 7. GUARDAR RESULTADOS
# =====================================================

high_df.to_csv("high_class_metrics_lisbon_2021_2023.csv", index=False)

print("\nArchivo 'high_class_metrics_lisbon_2021_2023.csv' guardado.")
