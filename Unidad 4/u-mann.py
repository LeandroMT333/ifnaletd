from scipy.stats import mannwhitneyu, norm
import numpy as np

def prueba_mann_whitney(grupo_a, grupo_b, alpha=0.05, alternativa='two-sided'):
    """
    Prueba U de Mann-Whitney con cálculo de Z y valores críticos.

    Parámetros:
    - grupo_a: lista o array de valores numéricos del grupo A
    - grupo_b: lista o array de valores numéricos del grupo B
    - alpha: nivel de significancia (por defecto 0.05)
    - alternativa: 'two-sided', 'less' o 'greater'

    Retorna:
    - Estadístico U
    - Valor p
    - Estadístico Z
    - Valor crítico Z
    - Interpretación del resultado
    """

    n1 = len(grupo_a)
    n2 = len(grupo_b)

    resultado = mannwhitneyu(grupo_a, grupo_b, alternative=alternativa)
    u = resultado.statistic
    p_valor = resultado.pvalue

    # Media y desviación estándar bajo H0
    mu_u = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    # Estadístico Z
    z = (u - mu_u) / sigma_u

    # Valores críticos según el tipo de prueba
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
    print(f"Estadístico U: {u}")
    print(f"Valor p: {p_valor:.5f}")
    print(f"Estadístico Z: {z:.5f}")
    print(f"Valor crítico Z (α = {alpha}): {z_crit:.5f}")

    if decision:
        print("Conclusión: Hay evidencia significativa para rechazar la hipótesis nula.")
        print("⇒ Se concluye que hay una diferencia entre el grupo A y el grupo B.")
    else:
        print("Conclusión: No hay evidencia suficiente para rechazar la hipótesis nula.")
        print("⇒ No se puede concluir que exista una diferencia entre los grupos.")

    return u, p_valor, z, z_crit

grupo_a = [12, 15, 14, 10, 13]
grupo_b = [22, 25, 24, 23, 26]

prueba_mann_whitney(grupo_a, grupo_b)
