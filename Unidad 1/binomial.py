import numpy as np
import scipy.stats as stats
from math import comb

def chi2_binomial_test(observed_counts, n=None, p=None, alpha=0.05):
    """
    Prueba Chi² de bondad de ajuste para distribución binomial.

    Parámetros:
    - observed_counts: lista de frecuencias observadas para k = 0, 1, ..., n
    - n: número de ensayos (si no se da, se asume len(observed_counts)-1)
    - p: probabilidad de éxito (si no se da, se estima)
    - alpha: nivel de significancia
    """
    k_values = np.arange(len(observed_counts))
    total = np.sum(observed_counts)

    if n is None:
        n = len(observed_counts) - 1

    # Estimar p si no se da
    if p is None:
        mean = np.sum(k_values * observed_counts) / total
        p = mean / n

    # Calcular frecuencias esperadas con la fórmula binomial
    expected_freqs = [comb(n, k) * (p**k) * ((1-p)**(n-k)) * total for k in k_values]

    # Agrupar clases con esperadas < 5
    obs_comb, exp_comb = [], []
    temp_obs, temp_exp = 0, 0

    for obs, exp in zip(observed_counts, expected_freqs):
        if exp < 5:
            temp_obs += obs
            temp_exp += exp
        else:
            if temp_exp > 0:
                obs_comb.append(temp_obs)
                exp_comb.append(temp_exp)
                temp_obs, temp_exp = 0, 0
            obs_comb.append(obs)
            exp_comb.append(exp)

    if temp_exp > 0:
        obs_comb.append(temp_obs)
        exp_comb.append(temp_exp)

    # Verificación de clases
    if len(obs_comb) < 2:
        print("No hay suficientes clases agrupadas con frecuencia esperada ≥ 5.")
        return

    # Estadístico y grados de libertad
    chi2_stat = np.sum((np.array(obs_comb) - np.array(exp_comb))**2 / np.array(exp_comb))
    df = len(obs_comb) - 1
    if p is None:
        df -= 1  # Si estimamos p, restamos 1

    # Valor crítico y p-valor
    chi2_crit = stats.chi2.ppf(1 - alpha, df)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    # Resultados
    print("--- Prueba Chi² de bondad de ajuste a distribución Binomial ---")
    print(f"n = {n}, p estimado = {p:.4f}")
    print(f"Estadístico Chi² calculado: {chi2_stat:.4f}")
    print(f"Grados de libertad: {df}")
    print(f"Valor crítico Chi² (α = {alpha}): {chi2_crit:.4f}")
    print(f"Valor p: {p_value:.4f}")

    if chi2_stat > chi2_crit:
        print("→ Se rechaza H₀: los datos NO siguen una distribución binomial.")
    else:
        print("→ No se rechaza H₀: los datos podrían seguir una distribución binomial.")


# === Ejemplo de uso ===
# Frecuencias observadas para k = 0 a 5
observadas = [12, 30, 40, 15, 2, 1]  # Total = 100
# Se puede pasar n=5 si se sabe, si no, se asume len(observadas) - 1
chi2_binomial_test(observadas)
