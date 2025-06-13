import numpy as np
from scipy import stats
import statsmodels.api as sm

def analizar_modelos(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    def imprimir_resultados(nombre, modelo, X):
        print(f"\nModelo {nombre}:")
        print(modelo.summary())

        r_squared = modelo.rsquared
        t_vals = modelo.tvalues
        p_vals = modelo.pvalues

        print(f"R²: {r_squared}")
        print(f"t (intercepto): {t_vals[0]}, p: {p_vals[0]}")
        for i in range(1, len(X[0])):
            print(f"t (coef x^{i}): {t_vals[i]}, p: {p_vals[i]}")

    print("=== REGRESIÓN LINEAL ===")
    # Lineal: y = a + b*x
    X_lineal = sm.add_constant(x)
    modelo_lineal = sm.OLS(y, X_lineal).fit()
    imprimir_resultados("Lineal", modelo_lineal, X_lineal)

    print("\n=== REGRESIÓN LOGARÍTMICA ===")
    # Logarítmica: y = a + b*ln(x)
    x_log = np.log(x)
    X_log = sm.add_constant(x_log)
    modelo_log = sm.OLS(y, X_log).fit()
    imprimir_resultados("Logarítmico", modelo_log, X_log)

    print("\n=== REGRESIÓN PARABÓLICA (CUADRÁTICA) ===")
    # Cuadrática: y = a + b1*x + b2*x²
    x2 = x**2
    X_parab = sm.add_constant(np.column_stack((x, x2)))
    modelo_parab = sm.OLS(y, X_parab).fit()
    imprimir_resultados("Parabólico", modelo_parab, X_parab)

    print("\n=== REGRESIÓN EXPONENCIAL ===")
    # Exponencial: y = a * e^(b*x) → ln(y) = ln(a) + b*x
    ln_y = np.log(y)
    X_exp = sm.add_constant(x)
    modelo_exp = sm.OLS(ln_y, X_exp).fit()
    imprimir_resultados("Exponencial", modelo_exp, X_exp)

    print("\n=== REGRESIÓN POTENCIAL ===")
    # Potencial: y = a*x^b → ln(y) = ln(a) + b*ln(x)
    ln_x = np.log(x)
    ln_y = np.log(y)
    X_pot = sm.add_constant(ln_x)
    modelo_pot = sm.OLS(ln_y, X_pot).fit()
    imprimir_resultados("Potencial", modelo_pot, X_pot)

# Ejemplo de uso
x = [1900
,1910
,1920
,1930
,1940
,1950
]
y = [75,
91.97,
105.7,
122.78,
131.7,
178.5
]
analizar_modelos(x, y)
