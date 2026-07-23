import numpy as np
import matplotlib.pyplot as plt

# Задача 1
x = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
y = np.array([4.2, 6.1, 8.5, 11.3, 16.1, 22.0, 36.0, 47.2, 64.2, 82.0])
n = len(x)

print(f"Количество точек: {n}")
print(f"x: {x}")
print(f"y: {y}")
print()

print("1. Квадратичная аппроксимация:")
sum_x = np.sum(x)
sum_x2 = np.sum(x**2)
sum_x3 = np.sum(x**3)
sum_x4 = np.sum(x**4)

sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2y = np.sum(x**2 * y)

A_quad = np.array([
    [n,      sum_x,   sum_x2],
    [sum_x,  sum_x2,  sum_x3],
    [sum_x2, sum_x3,  sum_x4]
])

b_quad = np.array([sum_y, sum_xy, sum_x2y])

coeffs_quad = np.linalg.solve(A_quad, b_quad)
a0, a1, a2 = coeffs_quad

y_quad = a0 + a1 * x + a2 * x**2
mse_quad = np.sqrt(np.mean((y - y_quad)**2))

print("Система уравнений (A * c = b):")
print("A =")
print(A_quad)
print("b =", b_quad)
print()
print(f"Решение: a0 = {a0:.6f}, a1 = {a1:.6f}, a2 = {a2:.6f}")
print(f"Уравнение: y = {a0:.4f} + {a1:.4f}*x + {a2:.4f}*x²")
print(f"СКО = {mse_quad:.6f}")
print()

print("2. Полиномиальная аппроксимация 4-й степени:")
sum_x5 = np.sum(x**5)
sum_x6 = np.sum(x**6)
sum_x7 = np.sum(x**7)
sum_x8 = np.sum(x**8)

sum_x3y = np.sum(x**3 * y)
sum_x4y = np.sum(x**4 * y)

A_poly4 = np.array([
    [n,      sum_x,   sum_x2,  sum_x3,  sum_x4],
    [sum_x,  sum_x2,  sum_x3,  sum_x4,  sum_x5],
    [sum_x2, sum_x3,  sum_x4,  sum_x5,  sum_x6],
    [sum_x3, sum_x4,  sum_x5,  sum_x6,  sum_x7],
    [sum_x4, sum_x5,  sum_x6,  sum_x7,  sum_x8]
])

b_poly4 = np.array([sum_y, sum_xy, sum_x2y, sum_x3y, sum_x4y])

coeffs_poly4 = np.linalg.solve(A_poly4, b_poly4)
b0, b1, b2, b3, b4 = coeffs_poly4

y_poly4 = b0 + b1*x + b2*x**2 + b3*x**3 + b4*x**4
mse_poly4 = np.sqrt(np.mean((y - y_poly4)**2))

print("Система уравнений (A * c = b):")
print("A =")
print(A_poly4)
print("b =", b_poly4)
print()
print(f"Решение: b0 = {b0:.6f}, b1 = {b1:.6f}, b2 = {b2:.6f}, b3 = {b3:.6f}, b4 = {b4:.6f}")
print(f"Уравнение: y = {b0:.4f} + {b1:.4f}*x + {b2:.4f}*x² + {b3:.4f}*x³ + {b4:.4f}*x⁴")
print(f"СКО = {mse_poly4:.6f}")
print()

print("3. Дробно-рациональная аппроксимация:")
M = np.column_stack([x, np.ones(n), -y])
d = x * y

A_rational = M.T @ M
b_rational = M.T @ d
coeffs_rational = np.linalg.solve(A_rational, b_rational)
a_r, b_r, c_r = coeffs_rational

y_rational = (a_r * x + b_r) / (x + c_r)
mse_rational = np.sqrt(np.mean((y - y_rational)**2))

print("Линеаризация: y*(x + c) = a*x + b")
print("следовательно a*x + b - c*y = x*y")
print()
print("Система уравнений (A * c = b):")
print("A =")
print(A_rational)
print("b =", b_rational)
print()
print(f"Решение: a = {a_r:.6f}, b = {b_r:.6f}, c = {c_r:.6f}")
print(f"Уравнение: y = ({a_r:.4f}*x + {b_r:.4f}) / (x + {c_r:.4f})")
print(f"СКО = {mse_rational:.6f}")
print()




print("Сравнение точности аппроксимации по среднеквадратичному отклонению:\n")
print(f"{'Метод':<30} {'СКО':<15}")
print("-" * 45)
print(f"{'Квадратичная':<30} {mse_quad:<15.6f}")
print(f"{'Полином 4-й степени':<30} {mse_poly4:<15.6f}")
print(f"{'Дробно-рациональная':<30} {mse_rational:<15.6f}")
print()

best_method = min([(mse_quad, "Квадратичная"), 
                   (mse_poly4, "Полином 4-й степени"), 
                   (mse_rational, "Дробно-рациональная")])
print(f"Наилучший метод: {best_method[1]} (СКО = {best_method[0]:.6f})")

# График
x_smooth = np.linspace(min(x), max(x), 100)

y_quad_smooth = a0 + a1*x_smooth + a2*x_smooth**2
y_poly4_smooth = b0 + b1*x_smooth + b2*x_smooth**2 + b3*x_smooth**3 + b4*x_smooth**4
y_rational_smooth = (a_r*x_smooth + b_r) / (x_smooth + c_r)

plt.figure(figsize=(12, 8))

plt.scatter(x, y, color='red', s=80, label='Исходные данные', zorder=5)
plt.plot(x_smooth, y_quad_smooth, 'b-', linewidth=2, label=f'Квадратичная (СКО={mse_quad:.4f})')
plt.plot(x_smooth, y_poly4_smooth, 'g-', linewidth=2, label=f'Полином 4 (СКО={mse_poly4:.4f})')
plt.plot(x_smooth, y_rational_smooth, 'orange', linewidth=2, label=f'Дробно-рациональная (СКО={mse_rational:.4f})')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Сравнение методов аппроксимации', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()





# Задача 2: аппроксимация Паде для ln(1+x)
print("\n\nПункт 2: аппроксимация Паде для ln(1+x)")

# Система для b1, b2, b3
A_pade = np.array([
    [1/3,  -1/2,   1],
    [-1/4,  1/3,  -1/2],
    [1/5,  -1/4,  1/3]
])

b_pade = np.array([1/4, -1/5, 1/6])

# Решаем систему
b1, b2, b3 = np.linalg.solve(A_pade, b_pade)

# Вычисляем a1, a2, a3
a1 = 1
a2 = b1 - 1/2
a3 = b2 - b1/2 + 1/3

print("Система уравнений для b1, b2, b3:")
print("A =")
print(A_pade)
print("b =", b_pade)
print()
print(f"Решение: b1 = {b1:.10f}, b2 = {b2:.10f}, b3 = {b3:.10f}")
print(f"a1 = {a1:.10f}")
print(f"a2 = {a2:.10f}")
print(f"a3 = {a3:.10f}")
print()
print(f"R₃,₃(x) = ({a1:.6f}*x + {a2:.6f}*x² + {a3:.6f}*x³) / (1 + {b1:.6f}*x + {b2:.6f}*x² + {b3:.6f}*x³)")
print()
# Исходная функция
def f_ln(x):
    return np.log(1 + x)

# Аппроксимация Паде
def pade_approximation(x):
    numerator = a1 * x + a2 * x**2 + a3 * x**3
    denominator = 1 + b1 * x + b2 * x**2 + b3 * x**3
    return numerator / denominator

# Точки для сравнения
x_pade = np.linspace(0, 1, 100)
y_true = f_ln(x_pade)
y_pade = pade_approximation(x_pade)

# СКО на отрезке [0, 1]
# Берем много точек для точной оценки
x_dense = np.linspace(0, 1, 1000)
y_true_dense = f_ln(x_dense)
y_pade_dense = pade_approximation(x_dense)
mse_pade = np.sqrt(np.mean((y_true_dense - y_pade_dense)**2))

print(f"СКО аппроксимации Паде на [0, 1]: {mse_pade:.10f}")
print()

# График
plt.figure(figsize=(10, 6))

plt.plot(x_pade, y_true, 'b-', linewidth=2.5, label='ln(1+x)')
plt.plot(x_pade, y_pade, 'r--', linewidth=2, label=f'R₃,₃(x) (СКО={mse_pade:.6f})')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Аппроксимация Паде R₃,₃ для ln(1+x)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
