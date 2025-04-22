import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, hamming_loss,
    classification_report, precision_score,
    recall_score, f1_score, confusion_matrix
)
from tensorflow.keras.models import load_model
from collections import Counter

# 1. Cargar modelo entrenado
modelo = load_model("modelo_procedimientos.h5")

# 2. Cargar datos originales
df = pd.read_csv("dataset_elpino.csv", sep=';', encoding='utf-8')

# 3. Dividir en secciones
df_diag = df.iloc[:, 0:35]
df_proc = df.iloc[:, 35:65]
edad = df['Edad en años']
df_sexo = df['Sexo (Desc)']

# 4. Codificar sexo
sexo_bin = df_sexo.map({'Hombre': 0, 'Mujer': 1}).fillna(0).astype(int)

# 5. Normalizar edad
edad_norm = MinMaxScaler().fit_transform(edad.values.reshape(-1, 1))

# 6. Codificar diagnósticos
codigos_diag = set()
for fila in df_diag.itertuples(index=False):
    for diag in fila:
        if pd.notnull(diag):
            codigos_diag.add(diag.split('-')[0].strip())
codigos_diag = sorted(codigos_diag)
dic_diag = {c: i for i, c in enumerate(codigos_diag)}

X_bin = np.zeros((len(df), len(dic_diag)), dtype=np.int8)
for i, fila in enumerate(df_diag.itertuples(index=False)):
    for diag in fila:
        if pd.notnull(diag):
            codigo = diag.split('-')[0].strip()
            if codigo in dic_diag:
                X_bin[i, dic_diag[codigo]] = 1

# 7. Preparar entradas finales
X = np.concatenate([X_bin, edad_norm, sexo_bin.values.reshape(-1, 1)], axis=1)

# 8. Codificar procedimientos (salida esperada)
codigos_proc = set()
for fila in df_proc.itertuples(index=False):
    for proc in fila:
        if pd.notnull(proc):
            codigos_proc.add(str(proc).split('-')[0].strip())
codigos_proc = sorted(codigos_proc)
dic_proc = {c: i for i, c in enumerate(codigos_proc)}

y = np.zeros((len(df), len(dic_proc)), dtype=np.int8)
for i, fila in enumerate(df_proc.itertuples(index=False)):
    for proc in fila:
        if pd.notnull(proc):
            codigo = str(proc).split('-')[0].strip()
            if codigo in dic_proc:
                y[i, dic_proc[codigo]] = 1

# 9. Predecir
predicciones = modelo.predict(X)
pred_bin = (predicciones > 0.5).astype(int)

# 10. Métricas generales
acc = accuracy_score(y, pred_bin)
hamming = hamming_loss(y, pred_bin)

print("\n===== RESULTADOS =====")
print(f"Accuracy global (exact match): {acc:.4f}")
print(f"Hamming Loss: {hamming:.4f} (mientras más bajo, mejor)")

# 11. Clasificación por etiqueta
print("\n===== METRICAS POR ETIQUETA =====")
print(classification_report(y, pred_bin, target_names=list(dic_proc.keys()), zero_division=0))

# 12. Métricas multilabel globales
print("\n===== MÉTRICAS GLOBALES MULTILABEL =====")
print(f"Micro Precision       : {precision_score(y, pred_bin, average='micro'):.4f}")
print(f"Micro Recall          : {recall_score(y, pred_bin, average='micro'):.4f}")
print(f"Micro F1-score        : {f1_score(y, pred_bin, average='micro'):.4f}")
print(f"Macro Precision       : {precision_score(y, pred_bin, average='macro'):.4f}")
print(f"Macro Recall          : {recall_score(y, pred_bin, average='macro'):.4f}")
print(f"Macro F1-score        : {f1_score(y, pred_bin, average='macro'):.4f}")
print(f"Weighted Precision    : {precision_score(y, pred_bin, average='weighted'):.4f}")
print(f"Weighted Recall       : {recall_score(y, pred_bin, average='weighted'):.4f}")
print(f"Weighted F1-score     : {f1_score(y, pred_bin, average='weighted'):.4f}")

# 13. Matriz de confusión para las 5 clases más frecuentes
print("\n===== MATRICES DE CONFUSIÓN (TOP 5 CLASES MÁS FRECUENTES) =====")

# Obtener top 5 clases con más ocurrencias reales
conteo = Counter()
for fila in y:
    for i, val in enumerate(fila):
        if val == 1:
            conteo[i] += 1
top5 = [idx for idx, _ in conteo.most_common(5)]

for i in top5:
    clase = list(dic_proc.keys())[i]
    print(f"\n>> Clase '{clase}'")
    print(confusion_matrix(y[:, i], pred_bin[:, i]))
