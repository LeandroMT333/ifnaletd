import numpy as np
import scipy.stats as stats

def chi2_poisson_test(observed_counts, alpha=0.05):
    """
    Realiza una prueba de bondad de ajuste Chi² para datos que se espera sigan
    una distribución de Poisson.
    
    Parámetros:
    - observed_counts: lista de frecuencias observadas para cada valor k (k=0,1,...)
    - alpha: nivel de significancia (por defecto 0.05)
    
    Asume que el índice de la lista representa el valor k observado.
    """
    # Valores observados y total de muestras
    k_values = np.arange(len(observed_counts))
    total = np.sum(observed_counts)
    
    # Estimar λ (media muestral)
    lambda_hat = np.sum(k_values * observed_counts) / total

    # Calcular frecuencias esperadas usando la función de masa de Poisson
    expected_probs = stats.poisson.pmf(k_values, mu=lambda_hat)
    expected_freqs = expected_probs * total

    # Combinar clases con frecuencias esperadas < 5
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

    # Agregar último grupo si quedó pendiente
    if temp_exp > 0:
        obs_comb.append(temp_obs)
        exp_comb.append(temp_exp)

    # Verificación
    if len(obs_comb) < 2:
        print("No hay suficientes clases agrupadas con frecuencia esperada ≥ 5.")
        return

    # Estadístico Chi² y grados de libertad
    chi2_stat = np.sum((np.array(obs_comb) - np.array(exp_comb))**2 / np.array(exp_comb))
    df = len(obs_comb) - 1 - 1  # -1 por estimar λ

    # Valor crítico y p-valor
    chi2_crit = stats.chi2.ppf(1 - alpha, df)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    # Mostrar resultados
    print(f"--- Prueba Chi² de bondad de ajuste a distribución Poisson ---")
    print(f"λ estimado: {lambda_hat:.4f}")
    print(f"Estadístico Chi² calculado: {chi2_stat:.4f}")
    print(f"Grados de libertad: {df}")
    print(f"Valor crítico Chi² (α = {alpha}): {chi2_crit:.4f}")
    print(f"Valor p: {p_value:.4f}")

    if chi2_stat > chi2_crit:
        print("→ Se rechaza la H₀: los datos NO siguen una distribución de Poisson.")
    else:
        print("→ No se rechaza la H₀: los datos podrían seguir una distribución de Poisson.")


# === Ejemplo de uso ===
# Observaciones para k = 0, 1, 2, ..., 6
observadas = [50, 80, 70, 40, 20, 5, 2]
chi2_poisson_test(observadas)
