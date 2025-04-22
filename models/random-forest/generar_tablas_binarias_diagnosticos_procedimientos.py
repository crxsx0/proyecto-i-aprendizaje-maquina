import pandas as pd

try:
    df_elpino = pd.read_csv('dataset_elpino.csv', sep=';', on_bad_lines='warn', engine='python')
except Exception as e:
    print(f"Error al leer el archivo: {e}")

# Cargar listas de códigos de procedimientos y diagnósticos desde los archivos Excel
df_proc_lits = pd.read_csv("codigos_procedimiento_unicos.csv")
codigos_proc = df_proc_lits["codigo"] 

df_diag_list = pd.read_csv("codigos_diagnostico_unicos.csv")
codigos_diag = df_diag_list["codigo"]

# Selección de las columnas de diagnóstico y procedimiento
df_diag = df_elpino.iloc[:, 0:35]
df_proc = df_elpino.iloc[:, 35:65]
df_sobra = df_elpino.iloc[:, 65:68]
df_sobra['Sexo_bin'] =  df_sobra['Sexo (Desc)'].map({'Hombre': 0, 'Mujer': 1})
df_sobra['Edad en años'] = pd.to_numeric(df_sobra['Edad en años'], errors='coerce')
df_sobra_rectificado = df_sobra[['Edad en años', 'Sexo_bin']]


# Crear listas con los códigos únicos de diagnóstico y procedimiento
lista_diag = codigos_diag.to_list()
lista_proc = codigos_proc.to_list()

# Crear DataFrame binario para diagnósticos
df_binario_diag = pd.DataFrame(columns=lista_diag)
diccionario_df_diag = dict.fromkeys(df_binario_diag.columns, 0)
filas_diccionario_diag = []

# Procesar los diagnósticos
for x in range(len(df_diag)):
    fila = df_diag.iloc[x]  # Obtener la fila de diagnóstico
    diccionario_df_diag = dict.fromkeys(df_binario_diag.columns, 0)
    for diag in fila:
        if pd.notnull(diag):  # Evitar errores con celdas vacías
            dig_part = diag.split("-")
            if dig_part != '':
                diccionario_df_diag[dig_part[0].strip()] = 1
    filas_diccionario_diag.append(diccionario_df_diag)
    if x % 500 == 0:
        print(f"Procesadas {x} filas de diagnóstico...")

# Guardar la tabla binaria de diagnósticos
df_binario_diag = pd.DataFrame(filas_diccionario_diag)

# Crear DataFrame binario para procedimientos
df_binario_proc = pd.DataFrame(columns=lista_proc)
diccionario_df_proc = dict.fromkeys(df_binario_proc.columns, 0)
filas_diccionario_proc = []

# Procesar los procedimientos
for x in range(len(df_proc)):
    fila = df_proc.iloc[x]  # Obtener la fila de procedimiento
    diccionario_df_proc = dict.fromkeys(df_binario_proc.columns, 0)
    for proc in fila:
        if pd.notnull(proc):  # Evitar errores con celdas vacías
            proc_part = proc.split("-")
            if proc_part != '':
                diccionario_df_proc[proc_part[0].strip()] = 1
    filas_diccionario_proc.append(diccionario_df_proc)

    if x % 500 == 0:
        print(f"Procesadas {x} filas de procedimientos...")

# Guardar la tabla binaria de procedimientos
df_binario_proc = pd.DataFrame(filas_diccionario_proc)

df_final_input = pd.concat([df_binario_diag, df_sobra_rectificado], axis = 1)
df_final_ouput = df_binario_proc
df_final_input = df_final_input.loc[:, ~df_final_input.columns.duplicated()]

# Mostrar resultados
df_final_input.to_csv('ElPinoBinarioinput.csv', sep= ";", index=False)
df_final_ouput.to_csv('ElPinoBinariooutput.csv', sep= ";", index=False)
print("Tablas binarias de diagnóstico y procedimiento guardadas correctamente.")