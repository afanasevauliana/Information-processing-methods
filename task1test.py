import numpy as np
import matplotlib.pyplot as plt

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
    
    for i in range(n):        # Вычисляем базисный полином ℓ_i(x)
        li = 1
        for j in range(n):
            if j != i:
                li *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        result += y_points[i] * li
    
    return result

def lagrange_barycentric(x_points, y_points, x_eval):
    n = len(x_points)
    
    # Шаг 1: Вычисляем веса w_i (один раз)
    weights = np.zeros(n)
    for i in range(n):
        w = 1.0
        for j in range(n):
            if j != i:
                w *= (x_points[i] - x_points[j])
        weights[i] = 1.0 / w
    
    # Шаг 2: Вычисляем значение в точке
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
    
    # Создаем таблицу n x n
    table = np.zeros((n, n))
    table[:, 0] = y_points  # Первый столбец - значения функции
    
    # Заполняем остальные столбцы
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (x_points[i+j] - x_points[i])
    
    # Коэффициенты - диагональ таблицы
    coeffs = table[0, :]  # Берем первую строку (это диагональ)
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

# 6. ВЫВОД СРАВНЕНИЯ МЕТОДОВ (ТОЛЬКО ТАБЛИЦА, БЕЗ ГРАФИКОВ)
def print_comparison_table(x_nodes, y_nodes, x_grid, y_true):
    """Выводит таблицу сравнения ошибок всех методов"""
    
    # Вычисляем все методы
    y_classic = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
    y_bary = np.array([lagrange_barycentric(x_nodes, y_nodes, x) for x in x_grid])
    coeffs, _ = divided_differences(x_nodes, y_nodes)
    y_newton = np.array([newton_polynomial(x_nodes, coeffs, x) for x in x_grid])
    
    # Вычисляем ошибки
    error_classic = np.max(np.abs(y_true - y_classic))
    error_bary = np.max(np.abs(y_true - y_bary))
    error_newton = np.max(np.abs(y_true - y_newton))
    
    # Выводим таблицу
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ МЕТОДОВ ИНТЕРПОЛЯЦИИ (МАКСИМАЛЬНЫЕ ОШИБКИ)")
    print("=" * 70)
    print(f"{'Метод':<30} | {'Максимальная ошибка':<20}")
    print("-" * 70)
    print(f"{'Лагранж (классический)':<30} | {error_classic:.2e}")
    print(f"{'Лагранж (барицентрический)':<30} | {error_bary:.2e}")
    print(f"{'Ньютон':<30} | {error_newton:.2e}")
    print("=" * 70)


def print_divided_differences_table(x_points, table):
    """Красиво выводит таблицу разделенных разностей"""
    n = len(x_points)
    
    print("\n" + "=" * 70)
    print("ТАБЛИЦА РАЗДЕЛЕННЫХ РАЗНОСТЕЙ")
    print("=" * 70)
    
    # Заголовок
    header = "x_i\tf(x_i)"
    for j in range(1, n):
        header += f"\t{j}-я разн."
    print(header)
    print("-" * 70)
    
    # Строки таблицы
    for i in range(n):
        row = f"{x_points[i]:.4f}\t{table[i, 0]:.6f}"
        for j in range(1, n - i):
            row += f"\t{table[i, j]:.6f}"
        print(row)
    
    print("-" * 70)
    # Выводим коэффициенты (диагональ)
    print("\nКоэффициенты для многочлена Ньютона (диагональ таблицы):")
    for i in range(n):
        print(f"  a{i} = {table[0, i]:.6f}")
    print("=" * 70)

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
        print("  0. Выход")
        
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
    
    print("\n" + "=" * 70)
    print(f"    ВЫБРАНА ФУНКЦИЯ: {func_name}")
    print(f"    Интервал: {interval}")
    print(f"    Количество узлов: {n}")





    while True:
        print("\nВыберите способ генерации узлов интерполяции:")
        print("1. Произвольное распределение (генерация случайных чисел)")
        print("2. Равномерное распределение")
        print("3. Узлы Чебышева")
        print("0. Выход")
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
        
        # Выводим узлы
        print(f"\nСгенерированы узлы:")
        print(f"   x = {np.array2string(x_nodes, precision=6)}")
        print(f"   y = {np.array2string(y_nodes, precision=6)}")
        
        # Создаем сетку для графиков
        x_grid = np.linspace(a, b, 200)
        y_true = f(x_grid)





        while True:
            print("Выберите, что хотите вычислить:")
            print("  1. Интерполяционный многочлен Лагранжа L(x) (классический способ)")
            print("  2. Интерполяционный многочлен Лагранжа L(x) (барицентрический вид)")
            print("  3. Разделенные разности")
            print("  4. Интерполяционный многочлен Ньютона N(x)")
            print("  5. Сравнить все методы")
            print("  0. Вернуться к выбору узлов")
            
            try:
                choice_method = int(input("Ваш выбор: "))
            except ValueError:
                print("Введите целое число")
                continue
            
            if choice_method == 0:
                print("\nВозврат к выбору узлов...")
                break
            
            if choice_method not in [1, 2, 3, 4, 5]:
                print("Выберите 1, 2, 3, 4, 5 или 0")
                continue

            if choice_method == 1:
                # 1. КЛАССИЧЕСКИЙ ЛАГРАНЖ
                print("\n" + "=" * 70)
                print("МЕТОД: Классический многочлен Лагранжа")
                print("=" * 70)
                
                print("⏳ Вычисление...")
                y_interp = np.array([lagrange_classic(x_nodes, y_nodes, x) for x in x_grid])
                
                # Вычисляем ошибку
                max_error = np.max(np.abs(y_true - y_interp))
                print(f"\nМаксимальная ошибка: {max_error:.2e}")
                
                # Выводим значения в нескольких точках
                print("\nЗначения в точках:")
                for x in test_points:
                    y_true_val = f(x)
                    y_interp_val = lagrange_classic(x_nodes, y_nodes, x)
                    print(f"   x = {x:.1f}: f(x) = {y_true_val:.6f}, L(x) = {y_interp_val:.6f}, ошибка = {abs(y_true_val - y_interp_val):.2e}")
            
            elif choice_method == 2:
                # 2. БАРИЦЕНТРИЧЕСКИЙ ЛАГРАНЖ
                print("\n" + "=" * 70)
                print("МЕТОД: Барицентрическая форма Лагранжа")
                print("=" * 70)
                
                print("Вычисление...")
                y_interp = np.array([lagrange_barycentric(x_nodes, y_nodes, x) for x in x_grid])
                
                max_error = np.max(np.abs(y_true - y_interp))
                print(f"\nМаксимальная ошибка: {max_error:.2e}")
                
                print("\nЗначения в точках:")
                for x in test_points:
                    y_true_val = f(x)
                    y_interp_val = lagrange_barycentric(x_nodes, y_nodes, x)
                    print(f"   x = {x:.1f}: f(x) = {y_true_val:.6f}, L(x) = {y_interp_val:.6f}, ошибка = {abs(y_true_val - y_interp_val):.2e}")
            
            elif choice_method == 3:
                # 3. РАЗДЕЛЕННЫЕ РАЗНОСТИ
                print("\n" + "=" * 70)
                print("МЕТОД: Разделенные разности")
                print("=" * 70)
                
                print("Вычисление...")
                coeffs, table = divided_differences(x_nodes, y_nodes)
                
                # Выводим таблицу
                print_divided_differences_table(x_nodes, table)
            
            elif choice_method == 4:
                # 4. МНОГОЧЛЕН НЬЮТОНА
                print("\n" + "=" * 70)
                print("МЕТОД: Многочлен Ньютона")
                print("=" * 70)
                
                print("Вычисление...")
                coeffs, table = divided_differences(x_nodes, y_nodes)
                
                # Выводим коэффициенты
                print("\nКоэффициенты многочлена Ньютона:")
                for i in range(len(coeffs)):
                    print(f"  a{i} = {coeffs[i]:.6f}")
                
                # Вычисляем значения
                y_interp = np.array([newton_polynomial(x_nodes, coeffs, x) for x in x_grid])
                
                max_error = np.max(np.abs(y_true - y_interp))
                print(f"\nМаксимальная ошибка: {max_error:.2e}")
                
                print("\nЗначения в точках:")
                for x in test_points:
                    y_true_val = f(x)
                    y_interp_val = newton_polynomial(x_nodes, coeffs, x)
                    print(f"   x = {x:.1f}: f(x) = {y_true_val:.6f}, N(x) = {y_interp_val:.6f}, ошибка = {abs(y_true_val - y_interp_val):.2e}")
            
            elif choice_method == 5:
                # 5. СРАВНЕНИЕ ВСЕХ МЕТОДОВ
                print("\n" + "=" * 70)
                print("СРАВНЕНИЕ ВСЕХ МЕТОДОВ ИНТЕРПОЛЯЦИИ")
                print("=" * 70)
                
                print("Вычисление всех методов...")
                print_comparison_table(x_nodes, y_nodes, x_grid, y_true)
            
            # Пауза перед возвратом в меню
            input("\nНажмите Enter, чтобы продолжить...")


if __name__ == "__main__":
    main()