import numpy as np
import statsmodels.api as sm

def regresion_multiple(y, *x_listas):
    """
    Realiza regresión lineal múltiple con análisis estadístico
    y: lista o array de la variable dependiente
    x_listas: múltiples listas o arrays de variables independientes
    """

    y = np.array(y, dtype=float)
    X_vars = [np.array(x, dtype=float) for x in x_listas]
    X = np.column_stack(X_vars)
    X_const = sm.add_constant(X)

    modelo = sm.OLS(y, X_const).fit()

    print("=== REGRESIÓN MÚLTIPLE ===")
    print(modelo.summary())

    print("\nCoeficientes:")
    for i, coef in enumerate(modelo.params):
        if i == 0:
            print(f"Intercepto (a): {coef}")
        else:
            print(f"Coeficiente x{i}: {coef}")

    print(f"\nR²: {modelo.rsquared}")
    
    print("\nEstadísticos t y p-valores:")
    for i, (t, p) in enumerate(zip(modelo.tvalues, modelo.pvalues)):
        nombre = "Intercepto" if i == 0 else f"x{i}"
        print(f"{nombre}: t = {t}, p = {p}")

    # Matriz de correlación entre y y cada x
    print("\nCoeficientes de correlación de Pearson entre cada xᵢ y y:")
    for i, x in enumerate(X_vars):
        r = np.corrcoef(x, y)[0, 1]
        print(f"r(x{i}, y): {r}")

# Ejemplo de uso
y = [10, 12, 13, 15, 18]
x1 = [1, 2, 3, 4, 5]
x2 = [2, 3, 5, 7, 11]

regresion_multiple(y, x1, x2)
