import numpy as np
import scipy.stats as stats

def chi2_test(observed, expected=None, alpha=0.05):
    """
    Realiza una prueba Chi-cuadrado (bondad de ajuste o independencia).
    
    Parámetros:
    - observed: lista o matriz de frecuencias observadas
    - expected: lista de frecuencias esperadas (solo para bondad de ajuste)
    - alpha: nivel de significancia (por defecto 0.05)

    Retorna:
    - estadísticas y decisión de la prueba
    """

    # Bondad de ajuste
    if expected is not None:
        chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        df = len(observed) - 1
        test_type = "Bondad de ajuste"
    # Independencia (tabla de contingencia)
    else:
        chi2_stat, p_value, df, _ = stats.chi2_contingency(observed)
        test_type = "Independencia"

    # Valor crítico
    chi2_crit = stats.chi2.ppf(1 - alpha, df)

    # Mostrar resultados
    print(f"--- Prueba Chi-cuadrado: {test_type} ---")
    print(f"Estadístico Chi² calculado: {chi2_stat:.4f}")
    print(f"Grados de libertad: {df}")
    print(f"Valor crítico Chi² (α = {alpha}): {chi2_crit:.4f}")
    print(f"Valor p: {p_value:.4f}")

    if chi2_stat > chi2_crit:
        print("→ Se rechaza la H₀ (diferencias significativas).")
    else:
        print("→ No se rechaza la H₀ (no hay diferencias significativas).")


# === Ejemplos de uso ===

# Ejemplo 1: Bondad de ajuste (se espera distribución uniforme)
observadas = [18, 22, 20, 25, 15]
esperadas = [20, 20, 20, 20, 20]
chi2_test(observadas, esperadas)

# Ejemplo 2: Independencia (tabla de contingencia)
tabla_contingencia = [
    [30, 10],
    [20, 40]
]
chi2_test(tabla_contingencia)
