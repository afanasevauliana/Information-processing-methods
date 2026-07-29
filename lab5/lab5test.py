from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Пример: X = ваши данные (frequency), y = шум (target)
X = data[['frequency']].values
y = data['noise'].values  # или как называется целевая переменная

# Создаем модель полиномиальной регрессии 2-й степени
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

# Обучаем
model.fit(X, y)

# Предсказания для визуализации
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model.predict(X_plot)

# Визуализация
plt.scatter(X, y, alpha=0.5, label='Данные')
plt.plot(X_plot, y_plot, color='red', label='Полиномиальная регрессия (степень 2)')
plt.xlabel('Frequency')
plt.ylabel('Noise')
plt.legend()
plt.title('Полиномиальная регрессия 2-й степени')
plt.show()