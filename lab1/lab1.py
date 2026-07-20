import numpy as np
import matplotlib.pyplot as plt
import time
import math

def investigate_lagrange_degrees(a, b, f, test_points):
    print("Исследование многочленов Лагранжа 1, 2, 3-й степени:")
    degrees = [1, 2, 3]  # Степени многочленов
    x_grid = np.linspace(a, b, 200)
    y_true = f(x_grid)
    
    results = []
    for deg in degrees:
        # Для степени deg нужно deg+1 узел
        n_nodes = deg + 1
        x_nodes = np.linspace(a, b, n_nodes)
        y_nodes = f(x_nodes)
        
        # Вычисляем интерполяцию
        y_lagrange = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
        max_error = np.max(np.abs(y_true - y_lagrange))
        results.append((deg, max_error))
        
        print(f"Степень {deg}: узлов = {n_nodes}, max_error = {max_error:.2e}")
    
    print(f"{'Степень':<10} | {'Максимальная ошибка':<20}")
    print("-" * 50)
    for deg, err in results:
        print(f"{deg:<10} | {err:.2e}")
    best = min(results, key=lambda x: x[1])
    print(f"\nЛучший результат: степень {best[0]} (ошибка = {best[1]:.2e})")

def theoretical_error(x_nodes, y_nodes, x_eval):
    # Теоретическая оценка погрешности.
    # |R_n(x)| <= M/(n+1)! * |Π(x - x_i)|
    n = len(x_nodes)
    M = 10.0
    
    product = 1.0
    for xi in x_nodes:
        product *= abs(x_eval - xi)
    
    return M / math.factorial(n + 1) * product

def investigate_n_impact(a, b, f, test_points, func_name):
    # Исследует влияние количества узлов на точность интерполяции.
    # Запускает интерполяцию для n = 3, 5, 7, 10, 15 и выводит результаты.
    print("Исследование влияния количества узлов на точность:")
    n_values = [3, 5, 7, 10, 15]
    results = []
    for n in n_values:
        # Генерируем равномерные узлы
        x_nodes = np.linspace(a, b, n)
        y_nodes = f(x_nodes)
        
        # Создаем сетку для проверки
        x_grid = np.linspace(a, b, 200)
        y_true = f(x_grid)
        
        # Вычисляем Лагранжа
        y_lagrange = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
        max_error = np.max(np.abs(y_true - y_lagrange))

        # Замер времени
        start = time.time()
        y_lagrange = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
        elapsed = time.time() - start
        
        results.append({
            'n': n,
            'max_error': max_error,
            'time': elapsed
        })
        
        print(f"n = {n:2d}: max_error = {max_error:.2e}, время = {elapsed:.6f} сек")

    print("\n" + "-" * 50)
    print(f"{'n':<6} | {'Максимальная ошибка':<20} | {'Время (сек)':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['n']:<6} | {r['max_error']:.2e}             | {r['time']:.6f}")
    
    return results


def investigate_runge_phenomenon():
    # Исследует феномен Рунге для g(x) = sin(x²) на [-π, π].
    # Сравнивает равномерные узлы и узлы Чебышева.
    print("Исследование феномена Рунге:")
    print("g(x) = sin(x²), x ∈ [-π, π]")
    a, b = -np.pi, np.pi
    def g(x):
        return np.sin(x**2)
    
    n = 15  # Количество узлов для исследования
    
    x_grid = np.linspace(a, b, 500)
    y_true = g(x_grid)
    
    # 1. Равномерные узлы
    print(f"\n1. Равномерные узлы (n = {n}):")
    x_uniform = np.linspace(a, b, n)
    y_uniform = g(x_uniform)

    y_lagrange_uniform = np.array([lagrange_classic(x_uniform, y_uniform, x) for x in x_grid])
    error_uniform = np.max(np.abs(y_true - y_lagrange_uniform))
    print(f"   Максимальная ошибка: {error_uniform:.2e}")
    
    # 2. Узлы Чебышева
    print(f"\n2. Узлы Чебышева (n = {n}):")
    x_cheb = np.array([np.cos((2*i + 1) / (2*n) * np.pi) for i in range(n)])
    y_cheb = g(x_cheb)
    
    y_lagrange_cheb = np.array([lagrange_classic(x_cheb, y_cheb, x) for x in x_grid])
    error_cheb = np.max(np.abs(y_true - y_lagrange_cheb))
    print(f"   Максимальная ошибка: {error_cheb:.2e}")
    
    # 3. Графики для визуализации
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Равномерные узлы
    ax1.plot(x_grid, y_true, 'b-', label='Исходная функция', linewidth=2)
    ax1.plot(x_uniform, y_uniform, 'ro', label='Узлы', markersize=6)
    ax1.plot(x_grid, y_lagrange_uniform, 'g--', label='Интерполяция', linewidth=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(f'Равномерные узлы (n={n})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # График 2: Узлы Чебышева
    ax2.plot(x_grid, y_true, 'b-', label='Исходная функция', linewidth=2)
    ax2.plot(x_cheb, y_cheb, 'ro', label='Узлы', markersize=6)
    ax2.plot(x_grid, y_lagrange_cheb, 'g--', label='Интерполяция', linewidth=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title(f'Узлы Чебышева (n={n})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # График 3: Ошибки для равномерных
    error_uniform_plot = np.abs(y_true - y_lagrange_uniform)
    ax3.plot(x_grid, error_uniform_plot, 'r-', label='Ошибка (равномерные)', linewidth=2)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('Ошибка при равномерных узлах')
    ax3.set_xlabel('x')
    ax3.set_ylabel('|f(x) - P(x)|')
    ax3.set_yscale('log')

    # График 4: Ошибки для Чебышева
    error_cheb_plot = np.abs(y_true - y_lagrange_cheb)
    ax4.plot(x_grid, error_cheb_plot, 'g-', label='Ошибка (Чебышева)', linewidth=2)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title('Ошибка при узлах Чебышева')
    ax4.set_xlabel('x')
    ax4.set_ylabel('|f(x) - P(x)|')
    ax4.set_yscale('log')
    
    plt.suptitle(f'Феномен Рунге: сравнение узлов (n={n})', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("")
    print("-" * 70)
    print(f"{'Тип узлов':<20} | {'Максимальная ошибка':<20}")
    print("-" * 70)
    print(f"{'Равномерные':<20} | {error_uniform:.2e}")
    print(f"{'Чебышева':<20} | {error_cheb:.2e}")
    
    if error_uniform > error_cheb * 10:
        print("\nВывод: Узлы Чебышева значительно уменьшают феномен Рунге")
    else:
        print("\nВывод: Узлы Чебышева помогают смягчить феномен Рунге") 

def generate_nodes(node_type, a, b, n):
    if node_type == 1:  # Случайные
        np.random.seed(42)
        return np.random.uniform(a, b, n)
    elif node_type == 2:  # Равномерные
        return np.linspace(a, b, n)
    elif node_type == 3:  # Чебышева
        return np.array([np.cos((2*i + 1) / (2*n) * np.pi) for i in range(n)])
    else:
        return None

def lagrange_classic(x_points, y_points, x_eval):
    n = len(x_points)
    result = 0
    
    for i in range(n): # Вычисляем базисный полином ℓ_i(x)
        li = 1
        for j in range(n):
            if j != i:
                li *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        result += y_points[i] * li
    
    return result

def lagrange_barycentric(x_points, y_points, x_eval):
    n = len(x_points)
    # 1: Вычисляем веса w_i (один раз)
    weights = np.zeros(n)
    for i in range(n):
        w = 1.0
        for j in range(n):
            if j != i:
                w *= (x_points[i] - x_points[j])
        weights[i] = 1.0 / w
    
    # 2: Вычисляем значение в точке
    # Проверяем, не попали ли в узел
    for i in range(n):
        if abs(x_eval - x_points[i]) < 1e-15:
            return y_points[i]
    numerator = 0
    denominator = 0
    for i in range(n):
        term = weights[i] / (x_eval - x_points[i])
        numerator += term * y_points[i]
        denominator += term
    
    return numerator / denominator

def divided_differences(x_points, y_points):
    n = len(x_points)
    table = np.zeros((n, n))
    table[:, 0] = y_points  # Первый столбец - значения функции
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (x_points[i+j] - x_points[i])
    coeffs = table[0, :]
    return coeffs, table

def newton_polynomial(x_points, coeffs, x_eval):
    #N(x) = a0 + a1*(x-x0) + a2*(x-x0)*(x-x1) + ...
    n = len(coeffs)
    result = coeffs[0]
    
    for i in range(1, n):
        term = coeffs[i]
        for j in range(i):
            term *= (x_eval - x_points[j])
        result += term
    
    return result

def print_comparison_table(x_nodes, y_nodes, x_grid, y_true): # Выводит таблицу сравнения ошибок всех методов
    # Вычисляем все методы
    y_classic = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
    y_bary = np.array([lagrange_barycentric(x_nodes, y_nodes, x) for x in x_grid])
    coeffs, _ = divided_differences(x_nodes, y_nodes)
    y_newton = np.array([newton_polynomial(x_nodes, coeffs, x) for x in x_grid])
    
    # Вычисляем ошибки
    error_classic = np.max(np.abs(y_true - y_classic))
    error_bary = np.max(np.abs(y_true - y_bary))
    error_newton = np.max(np.abs(y_true - y_newton))

    theoretical_errors = []
    for x in x_grid:
        theoretical_errors.append(theoretical_error(x_nodes, y_nodes, x))
    max_theoretical = np.max(theoretical_errors)
        
    print("\nСравнение всех методов интерполяции:")
    print("-" * 70)
    print(f"{'Метод':<30} | {'Практическая ошибка':<20} | {'Теоретическая':<15}")
    print("-" * 70)
    print(f"{'Лагранж (классический)':<30} | {error_classic:.2e}             | {max_theoretical:.2e}")
    print(f"{'Лагранж (барицентрический)':<30} | {error_bary:.2e}             | {max_theoretical:.2e}")
    print(f"{'Ньютон':<30} | {error_newton:.2e}             | {max_theoretical:.2e}")


def print_divided_differences_table(x_points, table):
    """Красиво выводит таблицу разделенных разностей"""
    n = len(x_points)
    
    print("\nТаблица разделенных разностей:")
    print(f"{'x_i':>10} | {'f(x_i)':>12} | ", end="")
    for j in range(1, n):
        print(f"{j}-я разн.{'':>6} | ", end="")
    print()
    print("-" * 99)
    
    for i in range(n):
        print(f"{x_points[i]:>10.4f} | ", end="")
        print(f"{table[i, 0]:>12.6f} | ", end="")
        for j in range(1, n - i):
            print(f"{table[i, j]:>15.6f} | ", end="")
        print()

    print("\nКоэффициенты для многочлена Ньютона (первая строка таблицы):")
    for i in range(n):
        if i == 0:
            print(f"  a0 = f[x0] = {table[0, 0]:.6f}")
        elif i == 1:
            print(f"  a1 = f[x0, x1] = {table[0, 1]:.6f}")
        elif i == 2:
            print(f"  a2 = f[x0, x1, x2] = {table[0, 2]:.6f}")
        else:
            indices = "".join([f"x{j}" for j in range(i + 1)])
            print(f"  a{i} = f[{', '.join([f'x{j}' for j in range(i + 1)])}] = {table[0, i]:.6f}")

    print("\nФормула многочлена Ньютона:")
    print("N(x) = a0 + a1·(x-x0) + a2·(x-x0)(x-x1) + a3·(x-x0)(x-x1)(x-x2) + ...")

def plot_results(x_grid, y_true, x_nodes, y_nodes, y_lagrange, y_newton, func_name, node_type_name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Исходная функция + узлы
    ax1.plot(x_grid, y_true, 'b-', label='Исходная функция', linewidth=2)
    ax1.plot(x_nodes, y_nodes, 'ro', label='Узлы интерполяции', markersize=8)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Исходная функция и узлы интерполяции')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # График 2: Лагранж
    ax2.plot(x_grid, y_true, 'b-', label='Исходная функция', linewidth=2, alpha=0.5)
    ax2.plot(x_grid, y_lagrange, 'g--', label='Лагранж', linewidth=2)
    ax2.plot(x_nodes, y_nodes, 'ro', label='Узлы', markersize=6)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Интерполяция Лагранжа')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # График 3: Ньютон
    ax3.plot(x_grid, y_true, 'b-', label='Исходная функция', linewidth=2, alpha=0.5)
    ax3.plot(x_grid, y_newton, 'm-.', label='Ньютон', linewidth=2)
    ax3.plot(x_nodes, y_nodes, 'ro', label='Узлы', markersize=6)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('Интерполяция Ньютона')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    
    # График 4: Ошибки
    error_lagrange = np.abs(y_true - y_lagrange)
    error_newton = np.abs(y_true - y_newton)
    ax4.plot(x_grid, error_lagrange, 'g-', label='Ошибка Лагранжа', linewidth=2)
    ax4.plot(x_grid, error_newton, 'm--', label='Ошибка Ньютона', linewidth=2)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title('Ошибки интерполяции')
    ax4.set_xlabel('x')
    ax4.set_ylabel('|f(x) - P(x)|')
    ax4.set_yscale('log')  # Логарифмическая шкала для наглядности
    
    plt.suptitle(f'{func_name}, узлы: {node_type_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    n = 5
    a, b = 0, 0
    f = None
    func_name = ""
    interval = ""
    test_points = []
    def f_arccos(x):
        return np.arccos(x)
    def g_runge(x):
        return np.sin(x**2)

    while True:
        print("Выберите функцию:")
        print("  1. f(x) = arccos(x), x ∈ [-1, 1] (основное задание)")
        print("  2. g(x) = sin(x²), x ∈ [-π, π] (феномен Рунге)")
        print("  0. Выход\n")
        
        try:
            choice_func = int(input("Ваш выбор: "))
        except ValueError:
            print("Введите целое число")
            continue
        
        if choice_func == 0:
            print("Программа завершена")
            return
        
        if choice_func == 1:
            a, b = -1, 1
            f = f_arccos
            func_name = "f(x) = arccos(x)"
            interval = f"[{a}, {b}]"
            test_points = [-0.5, 0, 0.5]  # Точки для проверки
            break

        elif choice_func == 2:
            a, b = -np.pi, np.pi
            f = g_runge
            func_name = "g(x) = sin(x²)"
            interval = f"[{a:.2f}, {b:.2f}]"
            test_points = [-2, 0, 2]  # Точки для проверки
            break
            
        else:
            print("Выберите 1, 2 или 0")
            continue
    
    print(f"Выбрана функция: {func_name}")
    print(f"Интервал: {interval}")
    print(f"Количество узлов: {n}")





    while True:
        print("\nВыберите способ генерации узлов интерполяции:")
        print("1. Произвольное распределение (генерация случайных чисел)")
        print("2. Равномерное распределение")
        print("3. Узлы Чебышева")
        print("0. Выход\n")
        try:
            choice = int(input("Ваш выбор: "))
        except ValueError:
            print("Введите целое число")
            continue
        if choice == 0:
            print("\nПрограмма завершена")
            break
        if choice not in [1, 2, 3]:
            print("Введите число от 0 до 3")
            continue
        x_nodes = generate_nodes(choice, a, b, n)
        print (f"Получившиеся узлы: {x_nodes}")
        y_nodes = f(x_nodes)
        
        print(f"\nСгенерированы узлы:")
        print(f"   x = {np.array2string(x_nodes, precision=6)}")
        print(f"   y = {np.array2string(y_nodes, precision=6)}")
        
        x_grid = np.linspace(a, b, 200)
        y_true = f(x_grid)





        while True:
            print("\nВыберите, что хотите вычислить:")
            print("  1. Интерполяционный многочлен Лагранжа L(x) (классический способ)")
            print("  2. Интерполяционный многочлен Лагранжа L(x) (барицентрический вид)")
            print("  3. Разделенные разности")
            print("  4. Интерполяционный многочлен Ньютона N(x)")
            print("  5. Сравнить все методы")
            print("  6. Построить графики")
            print("  7. Исследовать влияние количества узлов")
            print("  8. Исследовать феномен Рунге")
            print("  9. Сравнить типы узлов")
            print("  10. Полный анализ (все исследования)")
            print("  11. Исследовать многочлены Лагранжа 1, 2, 3-й степени")
            print("  0. Вернуться к выбору узлов\n")
            
            try:
                choice_method = int(input("Ваш выбор: "))
            except ValueError:
                print("Введите целое число")
                continue
            
            if choice_method == 0:
                print("\nВозврат к выбору узлов...")
                break
            
            if choice_method not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                print("Выберите 1-11 или 0")
                continue

            if choice_method == 1:
                print("\nМетод: Классический многочлен Лагранжа")
                
                y_lagrange = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
                coeffs, _ = divided_differences(x_nodes, y_nodes)
                y_newton = np.array([newton_polynomial(x_nodes, coeffs, x) for x in x_grid])
                
                max_error = np.max(np.abs(y_true - y_lagrange))
                print(f"\nМаксимальная ошибка: {max_error:.2e}")
                
                print("\nЗначения в точках:")
                for x in test_points:
                    y_true_val = f(x)
                    y_interp_val = lagrange_classic(x_nodes, y_nodes, x)
                    print(f"   x = {x:.1f}: f(x) = {y_true_val:.6f}, L(x) = {y_interp_val:.6f}, ошибка = {abs(y_true_val - y_interp_val):.2e}")
                
                plot_results(x_grid, y_true, x_nodes, y_nodes, y_lagrange, y_newton, func_name, "Лагранж (классический)")

            elif choice_method == 2:
                print("\nМетод: Барицентрическая форма Лагранжа")
                
                y_lagrange = np.array([lagrange_barycentric(x_nodes, y_nodes, x) for x in x_grid])
                coeffs, _ = divided_differences(x_nodes, y_nodes)
                y_newton = np.array([newton_polynomial(x_nodes, coeffs, x) for x in x_grid])
                
                max_error = np.max(np.abs(y_true - y_lagrange))
                print(f"\nМаксимальная ошибка: {max_error:.2e}")
                
                print("\nЗначения в точках:")
                for x in test_points:
                    y_true_val = f(x)
                    y_interp_val = lagrange_barycentric(x_nodes, y_nodes, x)
                    print(f"   x = {x:.1f}: f(x) = {y_true_val:.6f}, L(x) = {y_interp_val:.6f}, ошибка = {abs(y_true_val - y_interp_val):.2e}")
                
                plot_results(x_grid, y_true, x_nodes, y_nodes, y_lagrange, y_newton, func_name, "Лагранж (барицентрический)")
            
            elif choice_method == 3:
                # 3. РАЗДЕЛЕННЫЕ РАЗНОСТИ
                print("\nМетод: Разделенные разности")
                coeffs, table = divided_differences(x_nodes, y_nodes)
                
                # Выводим таблицу
                print_divided_differences_table(x_nodes, table)

            elif choice_method == 4:
                print("\nМетод: Многочлен Ньютона")
                coeffs, table = divided_differences(x_nodes, y_nodes)
                
                print("\nКоэффициенты многочлена Ньютона:")
                for i in range(len(coeffs)):
                    print(f"  a{i} = {coeffs[i]:.6f}")
                
                y_newton = np.array([newton_polynomial(x_nodes, coeffs, x) for x in x_grid])
                y_lagrange = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
                
                max_error = np.max(np.abs(y_true - y_newton))
                print(f"\nМаксимальная ошибка: {max_error:.2e}")
                
                print("\nЗначения в точках:")
                for x in test_points:
                    y_true_val = f(x)
                    y_interp_val = newton_polynomial(x_nodes, coeffs, x)
                    print(f"   x = {x:.1f}: f(x) = {y_true_val:.6f}, N(x) = {y_interp_val:.6f}, ошибка = {abs(y_true_val - y_interp_val):.2e}")
                
                plot_results(x_grid, y_true, x_nodes, y_nodes, y_lagrange, y_newton, func_name, "Ньютон")
            
            elif choice_method == 5:
                print_comparison_table(x_nodes, y_nodes, x_grid, y_true)

            elif choice_method == 6:
                y_lagrange = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
                coeffs, _ = divided_differences(x_nodes, y_nodes)
                y_newton = np.array([newton_polynomial(x_nodes, coeffs, x) for x in x_grid])
                plot_results(x_grid, y_true, x_nodes, y_nodes, y_lagrange, y_newton, func_name, "все методы")

            elif choice_method == 7:
                investigate_n_impact(a, b, f, test_points, func_name)

            elif choice_method == 8:
                investigate_runge_phenomenon()

            elif choice_method == 9:
                print("\nСравнение типов узлов:")
                
                node_types = [
                    (1, "Случайные"),
                    (2, "Равномерные"),
                    (3, "Чебышева")
                ]
                
                x_grid = np.linspace(a, b, 200)
                y_true = f(x_grid)
                
                results = []
                for node_type, type_name in node_types:
                    x_nodes = generate_nodes(node_type, a, b, n)
                    y_nodes = f(x_nodes)
                    y_lagrange = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
                    max_error = np.max(np.abs(y_true - y_lagrange))
                    results.append((type_name, max_error))
                    
                    print(f"{type_name}: max_error = {max_error:.2e}")
                
                best = min(results, key=lambda x: x[1])
                print(f"\nЛучший результат: {best[0]} (ошибка = {best[1]:.2e})")

            elif choice_method == 10:
                print("\nПолный анализ")

                print("\n1.")
                investigate_n_impact(a, b, f, test_points, func_name)
                
                print("\n2. Сравнение типов узлов:")
                x_grid = np.linspace(a, b, 200)
                y_true = f(x_grid)
                
                node_types = [
                    (1, "Случайные"),
                    (2, "Равномерные"),
                    (3, "Чебышева")
                ]
                
                for node_type, type_name in node_types:
                    x_nodes = generate_nodes(node_type, a, b, n)
                    y_nodes = f(x_nodes)
                    y_lagrange = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
                    max_error = np.max(np.abs(y_true - y_lagrange))
                    print(f"   {type_name}: {max_error:.2e}")
                
                print("\n3.")
                investigate_runge_phenomenon()

            elif choice_method == 11:
                investigate_lagrange_degrees(a, b, f, test_points)
            
            input("\nНажмите Enter, чтобы продолжить")


if __name__ == "__main__":
    main()