import matplotlib.pyplot as plt 
import numpy as np 
def lagrange_interpolation(x, xp, yp): # функции интерполяции для 1, 4, 6
    n = len(xp) 
    result = 0.0 
    for i in range(n): 
        term = yp[i]
        for j in range(n): 
            if i != j: 
                term *= (x - xp[j]) / (xp[i] - xp[j]) 
        result += term 
    return result 
def divided_differences(xp, yp): 
    n = len(xp)
    f = np.copy(yp) 
    for i in range(1, n): 
        for j in range(n - 1, i - 1, -1): 
            f[j] = (f[j] - f[j - 1]) / (xp[j] - xp[j - i]) 
    return f 
 
def print_error_table(func, a, b, node_type, ns=[3,5,10,15]): # таблица ошибок для пункта 6
    print(f"\n{'n':<5} {'Макс. ошибка Лагранжа':<25} {'Макс. ошибка Ньютона':<25}")
    print("-" * 60)
    for n in ns:
        if node_type == "uniform":
            xp = uniform_nodes(n, a, b)
        elif node_type == "chebyshev":
            xp = chebyshev_nodes(n, a, b)
        else:
            xp = random_nodes(n, a, b)
        xp = np.sort(xp)
        yp = func(xp)
        
        x_test = np.linspace(a, b, 200)
        y_true = func(x_test)
        
        y_lag = [lagrange_interpolation(xi, xp, yp) for xi in x_test]
        diffs = divided_differences(xp, yp)
        y_new = [newton_interpolation(xi, xp, yp, diffs) for xi in x_test]
        
        err_lag = np.max(np.abs(y_true - y_lag))
        err_new = np.max(np.abs(y_true - y_new))
        print(f"{n:<5} {err_lag:<25.6e} {err_new:<25.6e}")

def barycentric_weights(xp):
    n = len(xp)
    w = np.zeros(n)
    for i in range(n):
        w[i] = 1.0 / np.prod([xp[i] - xp[j] for j in range(n) if j != i])
    return w

def barycentric_lagrange(x, xp, yp, w):
    numerator = 0.0
    denominator = 0.0
    for i in range(len(xp)):
        if abs(x - xp[i]) < 1e-12:
            return yp[i]
        term = w[i] / (x - xp[i])
        numerator += term * yp[i]
        denominator += term
    return numerator / denominator

def newton_interpolation(x, xp, yp, divided_diffs): 
    n = len(xp) 
    result = divided_diffs[0] 
    for i in range(1, n): 
        term = divided_diffs[i] 
        for j in range(i): 
            term *= (x - xp[j]) 
        result += term 
    return result 
 
def f(x):
    return np.arccos(x)

def g(x):
    return np.sin(x**2) 
 
def random_nodes(n, a, b): 
    return np.random.uniform(a, b, n) 
 
def uniform_nodes(n, a, b): 
    return np.linspace(a, b, n) 
 
def chebyshev_nodes(n, a, b): 
    i = np.arange(1, n + 1) 
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i - 1) * np.pi / (2 * n)) 
 
def interpolate_and_plot(func, a, b, n, node_type="uniform"): 
    if node_type == "random": 
        x_points = random_nodes(n, a, b) 
    elif node_type == "uniform": 
        x_points = uniform_nodes(n, a, b) 
    elif node_type == "chebyshev": 
        x_points = chebyshev_nodes(n, a, b) 
 
    x_points = np.sort(x_points) 
    y_points = func(x_points) 
 
    x = np.linspace(a, b, 100) 
    y_true = func(x) 
    y_lagrange = [lagrange_interpolation(xi, x_points, y_points) for xi in x] 
 
    divided_diffs = divided_differences(x_points, y_points) 
    y_newton = [newton_interpolation(xi, x_points, y_points, divided_diffs) for xi in x] 

    plt.figure(figsize=(12, 6)) 
    plt.plot(x, y_true, label="Исходная функция", color="black") 
    plt.plot(x, y_lagrange, label="Интерполяция Лагранжа", linestyle="--", color="red") 
    plt.plot(x, y_newton, label="Интерполяция Ньютона", linestyle=":", color="blue") 
    plt.plot(x_points, y_points, 'o', label="Узлы интерполяции", color="green") 
 
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.title(f"Интерполяция функции с {node_type.capitalize()} узлами (n={n})") 
    plt.legend() 
    plt.grid(True) 
 
    plt.text(0.5, -0.1, f"Тип узлов: {node_type}", size=10, ha="center", transform=plt.gca().transAxes) 
    plt.show() 
 
    error_lagrange = np.abs(y_true - y_lagrange) 
    error_newton = np.abs(y_true - y_newton) 
 
    print(f"Макс. ошибка Лагранжа: {np.max(error_lagrange):.4f}") 
    print(f"Макс. ошибка Ньютона: {np.max(error_newton):.4f}") 
 
if __name__ == "__main__": 
 
    interpolate_and_plot(f, -1, 1, 5, node_type="uniform")
    interpolate_and_plot(f, -1, 1, 5, node_type="chebyshev")
    interpolate_and_plot(f, -1, 1, 10, node_type="uniform")   # для исследования увеличения узлов

    interpolate_and_plot(g, -np.pi, np.pi, 5, node_type="uniform")
    interpolate_and_plot(g, -np.pi, np.pi, 5, node_type="chebyshev")
    interpolate_and_plot(g, -np.pi, np.pi, 15, node_type="uniform")   # для феномена Рунге