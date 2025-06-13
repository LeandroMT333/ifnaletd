from scipy.stats import wilcoxon, norm
import numpy as np

def prueba_wilcoxon(pareados_a, pareados_b, alpha=0.05, alternativa='two-sided'):
    """
    Prueba de Wilcoxon para muestras pareadas.

    Parámetros:
    - pareados_a, pareados_b: listas o arrays de datos relacionados (antes/después, control/tratamiento)
    - alpha: nivel de significancia
    - alternativa: 'two-sided', 'less', 'greater'

    Retorna:
    - Estadístico W
    - Valor p
    - Estadístico Z (aproximado)
    - Valor crítico Z
    - Interpretación del resultado
    """

    # Validación de longitudes
    if len(pareados_a) != len(pareados_b):
        raise ValueError("Las muestras deben tener la misma longitud para la prueba de Wilcoxon.")

    resultado = wilcoxon(pareados_a, pareados_b, alternative=alternativa)
    w_stat = resultado.statistic
    p_valor = resultado.pvalue

    # Cálculo de Z aproximado (sólo válido si n es razonablemente grande)
    n = len(pareados_a)
    mu_w = n * (n + 1) / 4
    sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_stat - mu_w) / sigma_w

    # Valor crítico de Z
    if alternativa == 'two-sided':
        z_crit = norm.ppf(1 - alpha / 2)
        decision = abs(z) > z_crit
    elif alternativa == 'less':
        z_crit = norm.ppf(alpha)
        decision = z < z_crit
    elif alternativa == 'greater':
        z_crit = norm.ppf(1 - alpha)
        decision = z > z_crit
    else:
        raise ValueError("La alternativa debe ser 'two-sided', 'less' o 'greater'.")

    # Resultados
    print(f"Estadístico W (Wilcoxon): {w_stat}")
    print(f"Valor p: {p_valor:.5f}")
    print(f"Estadístico Z (aproximado): {z:.5f}")
    print(f"Valor crítico Z (α = {alpha}): {z_crit:.5f}")

    if decision:
        print("Conclusión: Hay evidencia significativa para rechazar la hipótesis nula.")
        print("⇒ Se concluye que hay una diferencia entre los pares de datos.")
    else:
        print("Conclusión: No hay evidencia suficiente para rechazar la hipótesis nula.")
        print("⇒ No se puede concluir que haya una diferencia significativa entre los pares.")

    return w_stat, p_valor, z, z_crit

antes = [85, 90, 88, 75, 95]
despues = [87, 91, 86, 78, 97]

prueba_wilcoxon(antes, despues, alpha=0.05, alternativa='two-sided')
