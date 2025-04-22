import pandas as pd

try:
    df_elpino = pd.read_csv('dataset_elpino.csv', sep=';', on_bad_lines='warn', engine='python')
except Exception as e:
    print(f"Error al leer el archivo: {e}")

df_proc_lits = pd.read_excel("CIE-9.xlsx")
codigos_proc = df_proc_lits["codigo"] 

df_diag_list= pd.read_excel("CIE-10.xlsx")
codigos_diag = df_diag_list["codigo"]

df_diag = df_elpino.iloc[:, 0:35]
df_proc = df_elpino.iloc[:, 35:65]
df_sobra = df_elpino.iloc[:, 65:68]

contador = 0
lista_diag = codigos_diag.to_list()
lista_malos_diag = []
for col in df_diag.columns:
    for diag in df_diag[col]:
        dig_part = diag.split("-")
        if (dig_part[0].strip() in lista_diag) == False and dig_part[0] != '' and dig_part[1] != '':
            lista_malos_diag.append(dig_part[0])
            contador += 1

contador = 0
lista_proc = codigos_proc.to_list()
lista_proc_limpia = [float(x) for x in lista_proc]
if 3.95 in lista_proc_limpia:
    print("paso")

df = pd.DataFrame(lista_proc_limpia, columns=['Valores'])
df.to_excel('salida.xlsx', index=False, engine='openpyxl')

lista_malos_proc = []
for col in df_proc.columns:
    for proc in df_proc[col]:
        proc_part = proc.split("-")
        if proc_part[0] == '':
            lista_malos_proc.append(proc_part[0])
            contador += 1
        elif (float(proc_part[0]) in lista_proc_limpia) == False and proc_part[0] != '' and proc_part[1] != '':
            lista_malos_proc.append(proc_part[0])
            contador += 1

print(set(lista_malos_diag))
print(set(lista_malos_proc))
