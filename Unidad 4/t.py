from scipy.stats import ttest_ind, t
import numpy as np

def prueba_t_independiente(grupo_a, grupo_b, alpha=0.05, alternativa='two-sided', equal_var=True):
    """
    Prueba t para muestras independientes con valor crítico y conclusión.

    Parámetros:
    - grupo_a, grupo_b: listas o arrays de datos numéricos
    - alpha: nivel de significancia
    - alternativa: 'two-sided', 'less', 'greater'
    - equal_var: True si se asumen varianzas iguales, False si no

    Retorna:
    - Estadístico t
    - Valor p
    - Valor crítico t
    - Interpretación del resultado
    """

    n1 = len(grupo_a)
    n2 = len(grupo_b)

    resultado = ttest_ind(grupo_a, grupo_b, alternative=alternativa, equal_var=equal_var)
    t_stat = resultado.statistic
    p_valor = resultado.pvalue

    # Grados de libertad
    if equal_var:
        gl = n1 + n2 - 2
    else:
        # Welch-Satterthwaite approximation
        s1 = np.var(grupo_a, ddof=1)
        s2 = np.var(grupo_b, ddof=1)
        gl = (s1/n1 + s2/n2)**2 / ((s1**2)/((n1**2)*(n1-1)) + (s2**2)/((n2**2)*(n2-1)))

    # Valor crítico
    if alternativa == 'two-sided':
        t_crit = t.ppf(1 - alpha / 2, df=gl)
        decision = abs(t_stat) > t_crit
    elif alternativa == 'less':
        t_crit = t.ppf(alpha, df=gl)
        decision = t_stat < t_crit
    elif alternativa == 'greater':
        t_crit = t.ppf(1 - alpha, df=gl)
        decision = t_stat > t_crit
    else:
        raise ValueError("La alternativa debe ser 'two-sided', 'less' o 'greater'.")

    print(f"Estadístico t: {t_stat:.5f}")
    print(f"Valor p: {p_valor:.5f}")
    print(f"Grados de libertad (df): {gl:.2f}")
    print(f"Valor crítico t (α = {alpha}): {t_crit:.5f}")

    if decision:
        print("Conclusión: Hay evidencia significativa para rechazar la hipótesis nula.")
        print("⇒ Se concluye que hay una diferencia entre los grupos.")
    else:
        print("Conclusión: No hay evidencia suficiente para rechazar la hipótesis nula.")
        print("⇒ No se puede concluir que exista una diferencia entre los grupos.")

    return t_stat, p_valor, gl, t_crit

grupo_a = [20, 22, 19, 23, 21]
grupo_b = [25, 27, 29, 26, 28]

prueba_t_independiente(grupo_a, grupo_b, alpha=0.05, alternativa='two-sided', equal_var=True)
