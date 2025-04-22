# IA para predecir procedimientos en base a diagnósticos, edad y sexo

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

# 1. Cargar CSV
df = pd.read_csv("dataset_elpino.csv", sep=';', encoding='utf-8')

# 2. Dividir en secciones
df_diag = df.iloc[:, 0:35]                      # Diagnósticos
df_proc = df.iloc[:, 35:65]                     # Procedimientos
edad = df['Edad en años']                     # Edad
df_sexo = df['Sexo (Desc)']                    # Sexo ('Hombre', 'Mujer')

# 3. Codificar sexo
sexo_bin = df_sexo.map({'Hombre': 0, 'Mujer': 1}).fillna(0).astype(int)

# 4. Normalizar edad
edad_norm = MinMaxScaler().fit_transform(edad.values.reshape(-1, 1))

# 5. Extraer códigos de diagnóstico
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

# 6. Concatenar entradas finales
X = np.concatenate([X_bin, edad_norm, sexo_bin.values.reshape(-1, 1)], axis=1)

# 7. Extraer códigos de procedimiento
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

# 8. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Crear modelo de red neuronal
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(dic_proc), activation='sigmoid')  # multietiqueta
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 10. Entrenar modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 11. Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión: {acc:.4f}")

# 12. Guardar modelo (opcional)
model.save("modelo_procedimientos.h5")
