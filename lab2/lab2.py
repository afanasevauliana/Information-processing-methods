import numpy as np
import matplotlib.pyplot as plt

def lagrange_classic(x_points, y_points, x_eval): # Классический многочлен Лагранжа
    n = len(x_points)
    result = 0
    for i in range(n):
        li = 1
        for j in range(n):
            if j != i:
                li *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        result += y_points[i] * li
    return result



def natural_cubic_spline(x_nodes, y_nodes): # Построение естественного кубического сплайна
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    
    # решаем систему для вторых производных M_i
    A = np.zeros((n+1, n+1))
    B = np.zeros(n+1)
    
    # краевые условия для естественного сплайна: M_0 = 0, M_n = 0
    A[0, 0] = 1
    A[n, n] = 1
    
    # внутренние уравнения
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 6 * ((y_nodes[i+1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i-1]) / h[i-1])
    
    M = np.linalg.solve(A, B)
    
    # коэффициенты сплайна
    a = y_nodes[:-1]
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        b[i] = (y_nodes[i+1] - y_nodes[i]) / h[i] - h[i] * (2 * M[i] + M[i+1]) / 6
        c[i] = M[i] / 2
        d[i] = (M[i+1] - M[i]) / (6 * h[i])
    
    return a, b, c, d, x_nodes

def eval_spline(spline_coeffs, x): # Вычисление значения сплайна в точке x
    a, b, c, d, x_nodes = spline_coeffs
    n = len(x_nodes) - 1
    
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i+1]:
            dx = x - x_nodes[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

    if x < x_nodes[0]:
        dx = x - x_nodes[0]
        return a[0] + b[0] * dx + c[0] * dx**2 + d[0] * dx**3
    else:
        dx = x - x_nodes[-1]
        i = n - 1
        return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

def eval_spline_vectorized(spline_coeffs, x_array): # Вычисление значений сплайна для массива точек
    return np.array([eval_spline(spline_coeffs, x) for x in x_array])

def compute_rmse(y_true, y_pred):
    """Вычисление среднеквадратичной ошибки (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Пункт 2 в файле: Сравнение полиномов и сплайнов

def compare_polynomial_spline(a, b, func, func_name, n_nodes_list, node_type_name="равномерные"):
    print(f"\nСравнение полиномов и сплайнов для {func_name}")
    print(f"Интервал: [{a:.2f}, {b:.2f}]")
    
    x_fine = np.linspace(a, b, 1000)
    y_true = func(x_fine)
    
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Сравнение полинома и сплайна для {func_name}', fontsize=14)
    for idx, n_nodes in enumerate(n_nodes_list):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Генерация узлов
        if node_type_name == "равномерные":
            x_nodes = np.linspace(a, b, n_nodes)
        elif node_type_name == "Чебышева":
            x_nodes = (a + b) / 2 + (b - a) / 2 * np.cos((2 * np.arange(1, n_nodes + 1) - 1) / (2 * n_nodes) * np.pi)
            x_nodes = np.sort(x_nodes)
        else:
            x_nodes = np.linspace(a, b, n_nodes)
        
        y_nodes = func(x_nodes)
        
        # Интерполяционный полином Лагранжа
        y_poly = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_fine])
        
        # Естественный кубический сплайн
        spline_coeffs = natural_cubic_spline(x_nodes, y_nodes)
        y_spline = eval_spline_vectorized(spline_coeffs, x_fine)
        
        # Вычисляем RMSE
        rmse_poly = compute_rmse(y_true, y_poly)
        rmse_spline = compute_rmse(y_true, y_spline)
        results.append((n_nodes, rmse_poly, rmse_spline))
        
        # Визуализация
        ax.plot(x_fine, y_true, 'k-', linewidth=2, label='Исходная функция')
        ax.plot(x_fine, y_poly, 'r--', linewidth=1.5, label=f'Полином (RMSE={rmse_poly:.2e})')
        ax.plot(x_fine, y_spline, 'b-', linewidth=1.5, label=f'Сплайн (RMSE={rmse_spline:.2e})')
        ax.plot(x_nodes, y_nodes, 'ko', markersize=5, label='Узлы')
        
        ax.set_title(f'n = {n_nodes}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод таблицы RMSE
    print("\nТаблица RMSE:")
    print(f"{'n':<8} | {'Полином':<15} | {'Сплайн':<15}")
    print("-" * 45)
    for n, rp, rs in results:
        print(f"{n:<8} | {rp:.2e}     | {rs:.2e}")
    
    # График сравнения RMSE
    plt.figure(figsize=(10, 6))
    n_vals = [r[0] for r in results]
    rmse_poly_vals = [r[1] for r in results]
    rmse_spline_vals = [r[2] for r in results]
    
    plt.semilogy(n_vals, rmse_poly_vals, 'ro-', linewidth=2, markersize=8, label='Полином Лагранжа')
    plt.semilogy(n_vals, rmse_spline_vals, 'bs-', linewidth=2, markersize=8, label='Естественный сплайн')
    plt.xlabel('Количество узлов интерполяции (n)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'Сравнение RMSE для {func_name}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results

# Пункт 3 в файле: Исследование скорости сходимости

def convergence_study(func, a, b, n_nodes_list, use_chebyshev=False):
    """
    Исследование скорости сходимости сплайнов
    """
    x_fine = np.linspace(a, b, 1000)
    y_true = func(x_fine)
    rmse_values = []
    
    for n_nodes in n_nodes_list:
        if use_chebyshev:
            x_nodes = (a + b) / 2 + (b - a) / 2 * np.cos((2 * np.arange(1, n_nodes + 1) - 1) / (2 * n_nodes) * np.pi)
            x_nodes = np.sort(x_nodes)
        else:
            x_nodes = np.linspace(a, b, n_nodes)
        
        y_nodes = func(x_nodes)
        spline_coeffs = natural_cubic_spline(x_nodes, y_nodes)
        y_spline = eval_spline_vectorized(spline_coeffs, x_fine)
        
        rmse = compute_rmse(y_true, y_spline)
        rmse_values.append(rmse)
    
    return rmse_values

def estimate_convergence_rate(n_nodes_list, rmse_values):
    """Оценка скорости сходимости по наклону прямой в логарифмическом масштабе"""
    log_n = np.log(n_nodes_list)
    log_rmse = np.log(rmse_values)
    slope, intercept = np.polyfit(log_n, log_rmse, 1)
    return -slope

def investigate_convergence(func, a, b, func_name, n_nodes_list):
    print(f"\nИсследование скорости сходимости сплайнов для {func_name}")
    print(f"Интервал: [{a:.2f}, {b:.2f}]")
    
    # Равномерные узлы
    rmse_uniform = convergence_study(func, a, b, n_nodes_list, use_chebyshev=False)
    
    # Узлы Чебышева
    rmse_cheb = convergence_study(func, a, b, n_nodes_list, use_chebyshev=True)
    
    # Вывод таблицы
    print("\nТаблица RMSE для сплайнов:")
    print(f"{'n':<8} | {'Равномерные узлы':<18} | {'Узлы Чебышева':<18}")
    print("-" * 50)
    for n, ru, rc in zip(n_nodes_list, rmse_uniform, rmse_cheb):
        print(f"{n:<8} | {ru:.2e}       | {rc:.2e}")
    
    # Оценка скорости сходимости
    rate_uniform = estimate_convergence_rate(n_nodes_list, rmse_uniform)
    rate_cheb = estimate_convergence_rate(n_nodes_list, rmse_cheb)
    
    print(f"\nЭкспериментальная скорость сходимости (равномерные узлы): {rate_uniform:.2f}")
    print(f"Экспериментальная скорость сходимости (узлы Чебышева): {rate_cheb:.2f}")
    print(f"Теоретическая оценка для кубического сплайна: 4")
    
    # График сходимости в логарифмическом масштабе
    plt.figure(figsize=(10, 6))
    plt.loglog(n_nodes_list, rmse_uniform, 'bo-', linewidth=2, markersize=8, label='Равномерные узлы')
    plt.loglog(n_nodes_list, rmse_cheb, 'rs-', linewidth=2, markersize=8, label='Узлы Чебышева')
    
    # Добавляем теоретическую оценку для сравнения (наклон -4)
    log_n = np.log(n_nodes_list)
    log_rmse_ref = np.log(rmse_uniform[0]) - 4 * (log_n - log_n[0])
    plt.loglog(n_nodes_list, np.exp(log_rmse_ref), 'k--', linewidth=1.5, label='Теоретическая оценка (O(h⁴))')
    
    plt.xlabel('Количество узлов интерполяции (n)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'Сходимость естественного кубического сплайна для {func_name}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return rmse_uniform, rmse_cheb

# =Пункт 4 в файле: Влияние выбора узлов

def investigate_node_choice(func, a, b, func_name, n_nodes):
    print(f"\nИсследование влияния выбора узлов для {func_name}")
    print(f"Интервал: [{a:.2f}, {b:.2f}], n = {n_nodes}")
    
    x_fine = np.linspace(a, b, 1000)
    y_true = func(x_fine)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Влияние выбора узлов на интерполяцию\n{func_name} на [{a:.2f}, {b:.2f}]', fontsize=14)
    node_types = [
        ("Равномерные", 1),
        ("Чебышева", 2)
    ]
    
    results = []
    
    for idx, (type_name, type_code) in enumerate(node_types):
        ax = axes[idx]
        if type_code == 1:  # Равномерные узлы
            x_nodes = np.linspace(a, b, n_nodes)
        else:  # Чебышева
            x_nodes = (a + b) / 2 + (b - a) / 2 * np.cos((2 * np.arange(1, n_nodes + 1) - 1) / (2 * n_nodes) * np.pi)
            x_nodes = np.sort(x_nodes)
        
        y_nodes = func(x_nodes)
        
        # Построение сплайна
        spline_coeffs = natural_cubic_spline(x_nodes, y_nodes)
        y_spline = eval_spline_vectorized(spline_coeffs, x_fine)
        
        # Вычисление ошибок
        rmse = compute_rmse(y_true, y_spline)
        max_error = np.max(np.abs(y_true - y_spline))
        results.append((type_name, rmse, max_error))
        
        # Визуализация
        ax.plot(x_fine, y_true, 'k-', linewidth=2, label='Исходная функция')
        ax.plot(x_fine, y_spline, 'b-', linewidth=1.5, label=f'Сплайн (RMSE={rmse:.2e})')
        ax.plot(x_nodes, y_nodes, 'ro', markersize=6, label='Узлы')
        ax.set_title(f'{type_name} узлы')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Вывод результатов
    print("\nРезультаты:")
    print(f"{'Тип узлов':<15} | {'RMSE':<12} | {'Максимальная ошибка':<20}")
    print("-" * 55)
    for type_name, rmse, max_err in results:
        print(f"{type_name:<15} | {rmse:.2e}     | {max_err:.2e}")
    
    best = min(results, key=lambda x: x[1])
    print(f"\nЛучший результат: {best[0]} узлы (RMSE = {best[1]:.2e})")
    
    plt.tight_layout()
    plt.show()
    
    return results

# ==================== ГЛАВНОЕ МЕНЮ ====================

def main():
    while True:
        print("\nВыберите функцию для исследования:")
        print("  1. f(x) = arccos(x), x ∈ [-1, 1] (основное задание)")
        print("  2. g(x) = sin(x²), x ∈ [-π, π] (феномен Рунге)")
        print("  0. Выход")
        
        try:
            choice_func = int(input("\nВаш выбор: "))
        except ValueError:
            print("Введите целое число")
            continue
        
        if choice_func == 0:
            print("Программа завершена")
            return
        
        if choice_func == 1:
            a, b = -1, 1
            func = np.arccos
            func_name = "f(x) = arccos(x)"
            break
        elif choice_func == 2:
            a, b = -np.pi, np.pi
            func = lambda x: np.sin(x**2)
            func_name = "g(x) = sin(x²)"
            break
        else:
            print("Выберите 1, 2 или 0")
    
    print(f"\nВыбрана функция: {func_name}")
    print(f"Интервал: [{a:.2f}, {b:.2f}]")
    
    # Основное меню
    while True:
        print("\nМеню:")
        print("  1. Сравнение интерполяционных полиномов и сплайнов")
        print("  2. Исследование скорости сходимости сплайнов")
        print("  3. Влияние выбора узлов интерполяции")
        print("  0. Выход")
        
        try:
            choice = int(input("\nВаш выбор: "))
        except ValueError:
            print("Введите целое число")
            continue
        
        if choice == 0:
            print("Программа завершена")
            break
        
        if choice == 1:
            # Задача 2
            n_nodes_list = [5, 10, 15, 20]
            print(f"\nИспользуемые количества узлов: {n_nodes_list}")
            
            # Сравнение с равномерными узлами
            compare_polynomial_spline(a, b, func, func_name, n_nodes_list, "равномерные")
            
        elif choice == 2:
            # Задача 3
            n_nodes_convergence = [5, 10, 20, 40, 80]
            print(f"\nИспользуемые количества узлов: {n_nodes_convergence}")
            investigate_convergence(func, a, b, func_name, n_nodes_convergence)
            
        elif choice == 3:
            # Задача 4
            n_nodes = 15
            print(f"\nКоличество узлов: {n_nodes}")
            investigate_node_choice(func, a, b, func_name, n_nodes)
            
        else:
            print("Выберите 0-3")
        
        input("\nНажмите Enter, чтобы продолжить")


if __name__ == "__main__":
    main()