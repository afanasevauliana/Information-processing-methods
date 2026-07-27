import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.pyplot as plt

# ============================================================
# 1. ДАННЫЕ (Вариант 3)
# ============================================================
# Скорость (м/с) и Производительность
x = np.array([13, 14, 16, 18, 19, 21, 23, 24, 26, 28, 29, 11, 19]).reshape(-1, 1)
y = np.array([11, 10, 12, 6, 5, 5, 12, 13, 16, 14, 31, 33, 19])

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №2: МЕТОДЫ РЕГРЕССИОННОГО АНАЛИЗА")
print("Вариант 3")
print("=" * 70)
print(f"Количество точек: {len(x)}")
print(f"x (Скорость): {x.flatten()}")
print(f"y (Производительность): {y}")
print()

# ============================================================
# ЗАДАНИЕ 1: ЛИНЕЙНАЯ РЕГРЕССИЯ
# ============================================================
print("=" * 70)
print("ЗАДАНИЕ 1: ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("=" * 70)

# Через формулу β = (X^T * X)^(-1) * X^T * y
X_design = np.column_stack([np.ones(len(x)), x])  # Добавляем столбец единиц
beta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
beta0, beta1 = beta

y_pred_linear = beta0 + beta1 * x.flatten()
mse_linear = mean_squared_error(y, y_pred_linear)

print(f"Уравнение: y = {beta0:.4f} + {beta1:.4f}*x")
print(f"Коэффициенты: β0 = {beta0:.4f}, β1 = {beta1:.4f}")
print(f"MSE = {mse_linear:.4f}")
print()

# Проверка через sklearn (для сравнения)
lr = LinearRegression()
lr.fit(x, y)
print(f"Проверка через sklearn: β0 = {lr.intercept_:.4f}, β1 = {lr.coef_[0]:.4f}")
print()

# ============================================================
# ЗАДАНИЕ 2: ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ (СТЕПЕНЬ 2)
# ============================================================
print("=" * 70)
print("ЗАДАНИЕ 2: ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ (СТЕПЕНЬ 2)")
print("=" * 70)

# Создаем полиномиальные признаки
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Обучаем модель
lr_poly = LinearRegression()
lr_poly.fit(x_poly, y)

# Предсказания
y_pred_poly = lr_poly.predict(x_poly)
mse_poly = mean_squared_error(y, y_pred_poly)

print(f"Коэффициенты: {lr_poly.intercept_:.4f}, {lr_poly.coef_[1]:.4f}, {lr_poly.coef_[2]:.4f}")
print(f"Уравнение: y = {lr_poly.intercept_:.4f} + {lr_poly.coef_[1]:.4f}*x + {lr_poly.coef_[2]:.4f}*x²")
print(f"MSE = {mse_poly:.4f}")
print()

# Сравнение с линейной
print(f"Сравнение MSE:")
print(f"  Линейная регрессия:     {mse_linear:.4f}")
print(f"  Полиномиальная (степень 2): {mse_poly:.4f}")
print(f"  Улучшение: {((mse_linear - mse_poly) / mse_linear * 100):.2f}%")
print()

# ============================================================
# ЗАДАНИЕ 3: РАЗБИЕНИЕ НА ОБУЧАЮЩУЮ/ТЕСТОВУЮ ВЫБОРКИ
# ============================================================
print("=" * 70)
print("ЗАДАНИЕ 3: РАЗБИЕНИЕ НА ОБУЧАЮЩУЮ/ТЕСТОВУЮ ВЫБОРКИ")
print("=" * 70)

# Разбиваем данные (70% / 30%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(f"Обучающая выборка: {len(x_train)} точек")
print(f"Тестовая выборка: {len(x_test)} точек")
print()

# Линейная регрессия на обучающей выборке
lr_train = LinearRegression()
lr_train.fit(x_train, y_train)
y_train_pred_linear = lr_train.predict(x_train)
y_test_pred_linear = lr_train.predict(x_test)
mse_train_linear = mean_squared_error(y_train, y_train_pred_linear)
mse_test_linear = mean_squared_error(y_test, y_test_pred_linear)

print("Линейная регрессия:")
print(f"  MSE (train) = {mse_train_linear:.4f}")
print(f"  MSE (test)  = {mse_test_linear:.4f}")
print()

# Полиномиальная регрессия на обучающей выборке
poly_train = PolynomialFeatures(degree=2)
x_train_poly = poly_train.fit_transform(x_train)
x_test_poly = poly_train.transform(x_test)

lr_poly_train = LinearRegression()
lr_poly_train.fit(x_train_poly, y_train)
y_train_pred_poly = lr_poly_train.predict(x_train_poly)
y_test_pred_poly = lr_poly_train.predict(x_test_poly)
mse_train_poly = mean_squared_error(y_train, y_train_pred_poly)
mse_test_poly = mean_squared_error(y_test, y_test_pred_poly)

print("Полиномиальная регрессия (степень 2):")
print(f"  MSE (train) = {mse_train_poly:.4f}")
print(f"  MSE (test)  = {mse_test_poly:.4f}")
print()

# Сравнительная таблица
print("Сравнительная таблица:")
print(f"{'Модель':<30} {'MSE (train)':<15} {'MSE (test)':<15}")
print("-" * 60)
print(f"{'Линейная':<30} {mse_train_linear:<15.4f} {mse_test_linear:<15.4f}")
print(f"{'Полиномиальная (степень 2)':<30} {mse_train_poly:<15.4f} {mse_test_poly:<15.4f}")
print()

# ============================================================
# ЗАДАНИЕ 4: RIDGE-РЕГРЕССИЯ. КРОСС-ВАЛИДАЦИЯ
# ============================================================
print("=" * 70)
print("ЗАДАНИЕ 4: RIDGE-РЕГРЕССИЯ. КРОСС-ВАЛИДАЦИЯ")
print("=" * 70)

# Создаем зашумленный признак x с шумом
np.random.seed(42)
noise = np.random.normal(0, 2, size=len(x))
x_noisy = x.flatten() + noise
x_noisy = x_noisy.reshape(-1, 1)

print(f"Добавлен шум к признаку x")
print(f"x (оригинальный): {x.flatten()}")
print(f"x (с шумом):      {x_noisy.flatten()}")
print()

# Полиномиальные признаки для зашумленных данных
poly_noisy = PolynomialFeatures(degree=2)
x_noisy_poly = poly_noisy.fit_transform(x_noisy)

# Список значений λ для тестирования
lambdas = [0.1, 1, 10]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"5-кратная кросс-валидация для Ridge-регрессии")
print(f"Значения λ: {lambdas}")
print()

results = []
for lam in lambdas:
    ridge = Ridge(alpha=lam)
    scores = cross_val_score(ridge, x_noisy_poly, y, cv=kf, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Переводим в положительные значения
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    results.append((lam, mean_mse, std_mse))
    
    # Обучаем модель на всех данных для получения коэффициентов
    ridge.fit(x_noisy_poly, y)
    y_pred_ridge = ridge.predict(x_noisy_poly)
    mse_ridge_full = mean_squared_error(y, y_pred_ridge)
    
    print(f"λ = {lam}:")
    print(f"  Средняя MSE (cross-val) = {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"  MSE (на всех данных) = {mse_ridge_full:.4f}")
    print(f"  Коэффициенты: {ridge.intercept_:.4f}, {ridge.coef_[1]:.4f}, {ridge.coef_[2]:.4f}")
    print()

# Выбираем лучший λ
best_lambda = min(results, key=lambda x: x[1])
print(f"Лучший λ: {best_lambda[0]} (MSE = {best_lambda[1]:.4f})")
print()

# Сравнение с обычной линейной регрессией (без регуляризации)
lr_noisy = LinearRegression()
lr_noisy.fit(x_noisy_poly, y)
y_pred_lr_noisy = lr_noisy.predict(x_noisy_poly)
mse_lr_noisy = mean_squared_error(y, y_pred_lr_noisy)

print("Сравнение на зашумленных данных:")
print(f"  Линейная регрессия (без регуляризации): MSE = {mse_lr_noisy:.4f}")
print(f"  Ridge (λ = {best_lambda[0]}): MSE = {best_lambda[1]:.4f}")
print()

# ============================================================
# ЗАДАНИЕ 5: ВИЗУАЛИЗАЦИЯ
# ============================================================
print("=" * 70)
print("ЗАДАНИЕ 5: ВИЗУАЛИЗАЦИЯ")
print("=" * 70)

# Создаем гладкие точки для графиков
x_smooth = np.linspace(min(x), max(x), 100).reshape(-1, 1)

# Для полиномиальной регрессии (степень 2)
poly_smooth = PolynomialFeatures(degree=2)
x_smooth_poly = poly_smooth.fit_transform(x_smooth)
y_smooth_poly = lr_poly.predict(x_smooth_poly)

# Для линейной регрессии
y_smooth_linear = beta0 + beta1 * x_smooth.flatten()

# Для Ridge на зашумленных данных (с лучшим λ)
ridge_best = Ridge(alpha=best_lambda[0])
ridge_best.fit(x_noisy_poly, y)
y_smooth_ridge = ridge_best.predict(x_smooth_poly)

# ===== ГРАФИК 1: Сравнение моделей регрессии =====
plt.figure(figsize=(12, 8))

plt.scatter(x, y, color='red', s=80, label='Исходные данные', zorder=5)
plt.plot(x_smooth, y_smooth_linear, 'b-', linewidth=2, label=f'Линейная (MSE={mse_linear:.4f})')
plt.plot(x_smooth, y_smooth_poly, 'g-', linewidth=2, label=f'Полиномиальная (MSE={mse_poly:.4f})')
plt.plot(x_smooth, y_smooth_ridge, 'orange', linewidth=2, label=f'Ridge (λ={best_lambda[0]}, MSE={best_lambda[1]:.4f})')

plt.xlabel('Скорость (м/с)', fontsize=12)
plt.ylabel('Производительность', fontsize=12)
plt.title('Сравнение моделей регрессии (Вариант 3)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===== ГРАФИК 2: Зависимость MSE от λ =====
plt.figure(figsize=(10, 6))

lambdas_plot = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
mse_plot = []

for lam in lambdas_plot:
    ridge = Ridge(alpha=lam)
    scores = cross_val_score(ridge, x_noisy_poly, y, cv=kf, scoring='neg_mean_squared_error')
    mse_plot.append(-np.mean(scores))

plt.semilogx(lambdas_plot, mse_plot, 'b-o', linewidth=2, markersize=8)
plt.axvline(x=best_lambda[0], color='red', linestyle='--', linewidth=2, label=f'Лучший λ = {best_lambda[0]}')
plt.xlabel('λ (логарифмическая шкала)', fontsize=12)
plt.ylabel('Средняя MSE (cross-validation)', fontsize=12)
plt.title('Зависимость MSE от параметра регуляризации λ', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===== ГРАФИК 3: Предсказания vs Реальные значения =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Линейная регрессия
axes[0].scatter(y, y_pred_linear, color='blue', s=50, alpha=0.7)
axes[0].plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=2)
axes[0].set_xlabel('Реальные значения')
axes[0].set_ylabel('Предсказанные значения')
axes[0].set_title(f'Линейная (MSE={mse_linear:.4f})')
axes[0].grid(True, alpha=0.3)

# Полиномиальная регрессия
axes[1].scatter(y, y_pred_poly, color='green', s=50, alpha=0.7)
axes[1].plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=2)
axes[1].set_xlabel('Реальные значения')
axes[1].set_ylabel('Предсказанные значения')
axes[1].set_title(f'Полиномиальная (MSE={mse_poly:.4f})')
axes[1].grid(True, alpha=0.3)

# Ridge-регрессия (на зашумленных данных)
ridge_plot = Ridge(alpha=best_lambda[0])
ridge_plot.fit(x_noisy_poly, y)
y_pred_ridge_plot = ridge_plot.predict(x_noisy_poly)
mse_ridge_plot = mean_squared_error(y, y_pred_ridge_plot)

axes[2].scatter(y, y_pred_ridge_plot, color='orange', s=50, alpha=0.7)
axes[2].plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=2)
axes[2].set_xlabel('Реальные значения')
axes[2].set_ylabel('Предсказанные значения')
axes[2].set_title(f'Ridge (λ={best_lambda[0]}, MSE={mse_ridge_plot:.4f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# ИТОГОВЫЙ ВЫВОД
# ============================================================
print("=" * 70)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 70)
print("1. Линейная регрессия:")
print(f"   Уравнение: y = {beta0:.4f} + {beta1:.4f}*x")
print(f"   MSE = {mse_linear:.4f}")
print()
print("2. Полиномиальная регрессия (степень 2):")
print(f"   Уравнение: y = {lr_poly.intercept_:.4f} + {lr_poly.coef_[1]:.4f}*x + {lr_poly.coef_[2]:.4f}*x²")
print(f"   MSE = {mse_poly:.4f}")
print()
print("3. Ridge-регрессия:")
print(f"   Лучший λ = {best_lambda[0]}")
print(f"   MSE (cross-validation) = {best_lambda[1]:.4f}")
print()
print("4. Сравнение на тестовой выборке (70/30):")
print(f"   Линейная:         MSE(train) = {mse_train_linear:.4f}, MSE(test) = {mse_test_linear:.4f}")
print(f"   Полиномиальная:   MSE(train) = {mse_train_poly:.4f}, MSE(test) = {mse_test_poly:.4f}")
print()
print("=" * 70)
print("РАБОТА ЗАВЕРШЕНА")
print("=" * 70)