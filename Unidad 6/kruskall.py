from scipy.stats import kruskal, chi2, rankdata

# Valores de entrada: puedes modificar estos grupos
grupo1 = [4, 5, 3, 3, 6, 1, 2, 2, 7, 5, 6, 4]
grupo2 = [1, 3, 2, 1, 1, 3, 4, 5]

# Lista de todos los grupos
grupos = [grupo1, grupo2]
nombres_grupos = [f"Grupo {i+1}" for i in range(len(grupos))]

# Parámetros
k = len(grupos)
n = sum(len(g) for g in grupos)
df = k - 1
alpha = 0.05

# Concatenar datos y asignar rangos
todos_los_datos = sum(grupos, [])
ranks = rankdata(todos_los_datos)

# Asignar rangos a cada grupo
rangos_por_grupo = []
inicio = 0
for grupo in grupos:
    fin = inicio + len(grupo)
    rangos = ranks[inicio:fin]
    rangos_por_grupo.append(rangos)
    inicio = fin

# Mostrar valores, rangos y suma de rangos de cada grupo
for nombre, grupo, rangos in zip(nombres_grupos, grupos, rangos_por_grupo):
    rangos_lista = [float(r) for r in rangos]  # convertir a float simple
    suma_rangos = sum(rangos_lista)
    print(f"\n{nombre} (n = {len(grupo)}):")
    print(f"  Valores: {grupo}")
    print(f"  Rangos:  {rangos_lista}")
    print(f"  Suma de rangos: {suma_rangos:.2f}")

# Prueba de Kruskal-Wallis
stat, p = kruskal(*grupos)

# Valor crítico
critical_value = chi2.ppf(1 - alpha, df)

# Tamaño del efecto (eta²)
eta_squared = (stat - k + 1) / (n - k)

# Resultados
print(f"\nEstadístico H de Kruskal-Wallis: {stat:.4f}")
print(f"Valor p: {p:.4f}")
print(f"Valor crítico para α = {alpha}: {critical_value:.4f}")
print(f"Tamaño del efecto (η²): {eta_squared:.4f}")

# Interpretaciones
if p < alpha:
    print("Resultado (valor p): Diferencias significativas entre los grupos.")
else:
    print("Resultado (valor p): No hay diferencias significativas entre los grupos.")

if stat > critical_value:
    print("Resultado (valor crítico): H > valor crítico → se rechaza H0.")
else:
    print("Resultado (valor crítico): H <= valor crítico → no se rechaza H0.")
