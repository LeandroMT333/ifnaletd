import numpy as np
from scipy.stats import norm

def runs_test_numericos(datos, alpha=0.05, referencia=None):
    """
    Prueba de rachas para secuencia numérica (aleatoriedad).
    
    Args:
        datos (list): Lista de valores numéricos.
        alpha (float): Nivel de significancia (por defecto 0.05).
        referencia (float or None): Umbral de referencia. Si es None, se usa la mediana.

    Returns:
        dict: Resultados estadísticos de la prueba.
    """
    datos = np.array(datos)

    # Usar la mediana como referencia si no se da otra
    if referencia is None:
        referencia = np.median(datos)

    # Convertir a secuencia binaria: 1 si >= referencia, 0 si < referencia
    binaria = np.where(datos >= referencia, 1, 0)

    # Contar rachas
    rachas = 1
    for i in range(1, len(binaria)):
        if binaria[i] != binaria[i - 1]:
            rachas += 1

    a = np.sum(binaria == 1)  # número de unos (≥ referencia)
    b = np.sum(binaria == 0)  # número de ceros (< referencia)

    if a == 0 or b == 0:
        return {
            "error": "La secuencia no tiene ambos grupos respecto al umbral; no se puede aplicar la prueba de rachas."
        }

    # Media y varianza esperadas
    media = (2 * a * b) / (a + b) + 1
    varianza = (2 * a * b * (2 * a * b - a - b)) / ((a + b)**2 * (a + b - 1))
    z = (rachas - media) / np.sqrt(varianza)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    z_critico = norm.ppf(1 - alpha / 2)

    return {
        "Referencia usada": referencia,
        "Secuencia binaria": binaria.tolist(),
        "Total de datos": len(datos),
        "Número de unos (≥ ref)": int(a),
        "Número de ceros (< ref)": int(b),
        "Número de rachas observadas": rachas,
        "Media esperada de rachas": media,
        "Varianza esperada": varianza,
        "Z calculado": z,
        "Z crítico": z_critico,
        "p-value": p_value,
        "Conclusión": "Aleatoria" if p_value > alpha else "No aleatoria"
    }

# Ejemplo de uso:
if __name__ == "__main__":
    datos = [13, 17, 15, 12, 14, 11, 10, 18, 16, 12]
    resultado = runs_test_numericos(datos)

    for k, v in resultado.items():
        print(f"{k}: {v}")
