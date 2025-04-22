import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib

# === 1. Cargar los datos ===
df_input = pd.read_csv("ElPinoBinarioinput.csv", sep=';', engine='python')
df_output = pd.read_csv("ElPinoBinariooutput.csv", sep=';', engine='python')

# === 2. Preprocesar ===
X = df_input.fillna(0)
y = df_output.fillna(0)

# === 3. Dividir en train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Entrenar manualmente con progreso ===
modelos = []
print("Entrenando modelos individuales con barra de progreso:")
for i in tqdm(range(y_train.shape[1]), desc="Entrenando etiquetas"):
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train.iloc[:, i])
    modelos.append(clf)

# === 5. Predicción ===
y_pred = pd.DataFrame({
    y_train.columns[i]: modelos[i].predict(X_test)
    for i in range(len(modelos))
})

# === 6. Evaluación ===
print("\nEvaluación general:")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(modelos, "random_forest_modelos.pkl")