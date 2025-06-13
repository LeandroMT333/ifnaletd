from scipy.stats import norm

# Datos de ejemplo
datos = [12, 15, 14, 10, 11, 13, 17, 16, 12, 14]

# Hipótesis nula: mediana esperada
mediana_esperada = 13
alpha = 0.05  # Nivel de significancia

# Cálculo de signos
positivos = sum(x > mediana_esperada for x in datos)
negativos = sum(x < mediana_esperada for x in datos)
empates = len(datos) - (positivos + negativos)

# Estadísticos clave
x = min(positivos, negativos)  # éxitos en prueba binomial
n = positivos + negativos      # sin empates

# Parámetros de la prueba normal aproximada (Z)
media = n * 0.5
desviacion = (n * 0.5 * 0.5) ** 0.5

# Z calculado con corrección de continuidad
z = (x - media + 0.5) / desviacion if x < media else (x - media - 0.5) / desviacion

# Valor crítico Z para prueba bilateral
z_critico = norm.ppf(1 - alpha / 2)

# Valor p
p_value = 2 * norm.cdf(-abs(z))

# Mostrar resultados
print("=== Prueba de Signos ===")
print(f"Positivos: {positivos}")
print(f"Negativos: {negativos}")
print(f"Empates (0): {empates}")
print(f"x (menor de positivos/negativos): {x}")
print(f"n (sin empates): {n}")
print(f"z calculado: {z:.4f}")
print(f"z crítico (α={alpha}): ±{z_critico:.4f}")
print(f"p-value: {p_value:.4f}")
print("Conclusión:", "Diferencia significativa" if abs(z) > z_critico else "No se rechaza H0 (sin diferencia)")
