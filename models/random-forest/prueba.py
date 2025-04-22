import pandas as pd

# Leer el CSV
df = pd.read_csv("codigos_procedimiento_unicos.csv")

# Seleccionar la columna (ejemplo: "columna1")
columna = df["codigo"]

# Obtener el valor más repetido
valor_mas_comun = columna.mode()[0]  # Si hay empate, toma el primero

# También puedes ver cuántas veces aparece
frecuencia = columna.value_counts().iloc[0]

print(f"El valor más repetido es: {valor_mas_comun} (se repite {frecuencia} veces)")
