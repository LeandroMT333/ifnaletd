import numpy as np
from scipy.stats import norm

def runs_test(sequence, alpha=0.05):
    """
    Realiza la prueba de rachas (runs test) para evaluar aleatoriedad.
    
    Args:
        sequence (list or array): Lista de valores binarios (0/1) o cualquier valor numérico.
        alpha (float): Nivel de significancia para la prueba (default=0.05).

    Returns:
        dict: Resultados estadísticos de la prueba.
    """
    # Si no es binaria, convertir a secuencia de arriba/abajo respecto a la media
    sequence = np.array(sequence)
    if not set(sequence).issubset({0, 1}):
        median = np.median(sequence)
        sequence = np.where(sequence >= median, 1, 0)

    # Contar número de rachas
    runs = 1  # al menos una racha
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            runs += 1

    n1 = np.sum(sequence == 1)
    n0 = np.sum(sequence == 0)

    if n1 == 0 or n0 == 0:
        return {
            "error": "La secuencia no contiene ambos valores, no se puede aplicar la prueba de rachas."
        }

    # Cálculo de media y varianza esperada de las rachas
    mean_runs = (2 * n1 * n0) / (n1 + n0) + 1
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / \
               ((n1 + n0) ** 2 * (n1 + n0 - 1))
    
    # Estadístico Z
    z = (runs - mean_runs) / np.sqrt(var_runs)

    # Valor p (dos colas)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    result = {
        "n1 (unos)": int(n1),
        "n0 (ceros)": int(n0),
        "Número de rachas observadas": runs,
        "Media esperada": mean_runs,
        "Varianza esperada": var_runs,
        "Estadístico Z": z,
        "Valor p": p_value,
        "Conclusión": "Aleatoria" if p_value > alpha else "No aleatoria",
        "Nivel de significancia": alpha
    }

    return result

# Ejemplo de uso:
if __name__ == "__main__":
    # Puedes probar con una secuencia binaria o numérica
    secuencia = [0, 0, 0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    resultados = runs_test(secuencia)
    
    for k, v in resultados.items():
        print(f"{k}: {v}")
