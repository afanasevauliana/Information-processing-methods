import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

speed = np.array([13, 14, 16, 18, 19, 21, 23, 24, 26, 28, 29, 31, 33])
performance = np.array([11, 10, 12, 6, 5, 5, 12, 13, 16, 14, 15, 11, 19])
n = len(speed)

print("\nЗадание 1: Линейная регрессия (y = b0 + b1*x)")

X = np.column_stack([np.ones(n), speed])  # Матрица (13x2)
y = performance.reshape(-1, 1)
print("Матрица X (чтобы учесть b0, добавлен столбец из единиц):")
print(X)

#2. Коэффициенты
# Формула: b = (X^T * X)^(-1) * X^T * y
XTX = X.T @ X          # X^T * X  (2x2)
XTX_inv = np.linalg.inv(XTX)  # (X^T * X)^(-1)
XTy = X.T @ y          # X^T * y  (2x1)
beta = XTX_inv @ XTy   # b = (X^T * X)^(-1) * X^T * y
b0, b1 = beta[0][0], beta[1][0]
print("\nМатрица X^T * X:")
print(XTX)
print()
print("Обратная матрица (X^T * X)^(-1):")
print(XTX_inv)
print()
print(f"Коэффициенты:")
print(f"b0 = {b0:.6f}")
print(f"b1 = {b1:.6f}")
print(f"Уравнение: y = {b0:.4f} + {b1:.4f} * x")
print()

# 3. Предсказанные значения и MSE
y_pred = X @ beta
mse = np.mean((y - y_pred)**2)
print(f"Уравнение: y = {b0:.4f} + {b1:.4f} * x")
print("Предсказанные значения:")
print(y_pred.flatten())
print()
print(f"MSE = {mse:.6f}")
print("\n------------------------------------------------------------------------------------------------\n")





print("Задание 2: Полиномиальная регрессия степени 2 (y = b0 + b1*x + b2*x²):")
# 1. Формируем матрицу X: столбцы [1, x, x²]
X_poly2 = np.column_stack([np.ones(n), speed, speed**2])  # Матрица (13x3)
y = performance.reshape(-1, 1)  # Вектор-столбец (13x1)

print("Матрица X:")
print(X_poly2)
print()

# 2. Коэффициенты по формуле: b = (X^T * X)^(-1) * X^T * y
XTX2 = X_poly2.T @ X_poly2          # X^T * X  (3x3)
XTX_inv2 = np.linalg.inv(XTX2)      # (X^T * X)^(-1)
XTy2 = X_poly2.T @ y                # X^T * y  (3x1)
beta2 = XTX_inv2 @ XTy2             # b = (X^T * X)^(-1) * X^T * y
b0_2, b1_2, b2_2 = beta2[0][0], beta2[1][0], beta2[2][0]

print("Матрица X^T * X:")
print(XTX2)
print()
print(f"Коэффициенты:")
print(f"b0 = {b0_2:.6f}")
print(f"b1 = {b1_2:.6f}")
print(f"b2 = {b2_2:.6f}")
print(f"Уравнение: y = {b0_2:.4f} + {b1_2:.4f}*x + {b2_2:.4f}*x²")
print()

# 3. Предсказанные значения и MSE
y_pred2 = X_poly2 @ beta2
mse2 = np.mean((y - y_pred2)**2)
print("Предсказанные значения:")
print(y_pred2.flatten())
print()
print(f"MSE = {mse2:.6f}")

if mse < mse2:
    print("Полиномиальная регрессия степени 2 лучше (MSE меньше)")
else:
    print("Линейная регрессия лучше (MSE меньше)")
print("\n------------------------------------------------------------------------------------------------\n")







# Зашумлённый признак
np.random.seed(42)
noise = np.random.normal(0, 2, n)
speed_noisy = speed + noise

X = np.column_stack([speed, speed_noisy])
y = performance

print("\nЗадание 3: Ridge-регрессия с кросс-валидацией")
print(f"speed: {speed}")
print(f"speed_noisy: {speed_noisy}")
print()

# Ручное разбиение на 5 частей (folds)
def kfold_split(X, y, n_splits=5, shuffle=True, random_state=42):
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
    fold_size = n // n_splits
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_indices, test_indices))
    return folds

lambdas = [0.1, 1, 10]
results = {}
best_mse = float('inf')
best_lambda = None

folds = kfold_split(X, y, n_splits=5, shuffle=True, random_state=42)

for lam in lambdas:
    mse_scores = []
    coefs_list = []
    
    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        ridge = Ridge(alpha=lam)
        ridge.fit(X_train, y_train)
        
        # Предсказания и MSE
        y_pred = ridge.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        coefs_list.append(ridge.coef_)
    
    mean_mse = np.mean(mse_scores)
    mean_coefs = np.mean(coefs_list, axis=0)
    
    results[lam] = {
        'mean_mse': mean_mse,
        'coefs': mean_coefs,
    }
    
    if mean_mse < best_mse:
        best_mse = mean_mse
        best_lambda = lam

print("Сравнение результатов:")
print(f"{'Лямбда':<10} {'Среднее MSE':<15} {'Коэффициенты (b1, b2)'}")
print("----------------------------------------------------")
for lam in lambdas:
    print(f"{lam:<10} {results[lam]['mean_mse']:<15.6f} {results[lam]['coefs']}")
print()
print(f"Лучшая лямбда: {best_lambda} (среднее MSE = {best_mse:.6f})")