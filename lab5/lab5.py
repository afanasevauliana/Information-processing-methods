import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


data = pd.read_csv('train.csv')
print("Первые 5 строк данных:")
print(data.head())
print(f"\nРазмер датасета: {data.shape[0]} строки и {data.shape[1]} столбцов")
print("\nТипы данных:")
print(data.dtypes)
print("\nПроверка пропусков:")
print(data.isnull().sum())
print(f"\nКол-во дубликатов: {data.duplicated().sum()}")
#  heatmap пропусков
sns.heatmap(data.isnull(), cbar=False)
plt.title('Распределение пропусков')
plt.show()






numeric_cols = data.columns
# Гистограмма + KDE для числовых признаков
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col], kde=True)
    plt.title(f'Распределение {col}')
    plt.ylabel('Частота')
plt.tight_layout()
plt.show()
# Boxplot
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=data[col])
    plt.title(f'Boxplot {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()
# Корреляции
print("\nТаблица корреляций:")
print(data.corr())







# data	Исходные данные (из файла)
# data_before	Данные ДО удаления выбросов (для боксплотов)
# data_clean	Данные ПОСЛЕ удаления выбросов
# data_before_log	Данные ДО логарифмирования (для гистограмм)
# data_encoded	Финальные данные (после всей предобработки)
data_clean = data.copy()
print("\n----------------До предобработки:----------------")
print(f"Пропусков: {data_clean.isnull().sum().sum()}")
print(f"Размер: {data_clean.shape}")

data_before = data.copy()

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

for col in data_clean.columns:
    data_clean = remove_outliers_iqr(data_clean, col)
print("\nОбработка выбросов проведена методом IQR")
print(f"Размер после удаления выбросов: {data_clean.shape}")

# Боксплоты ДО и ПОСЛЕ
plt.figure(figsize=(15, 10))
for i, col in enumerate(data.columns, 1):
    # ДО
    plt.subplot(2, 6, i)
    sns.boxplot(y=data_before[col])
    plt.title(f'ДО: {col}')
    plt.ylabel(col)
    # ПОСЛЕ
    plt.subplot(2, 6, i + 6)
    sns.boxplot(y=data_clean[col])
    plt.title(f'ПОСЛЕ: {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

print("\n---------Логарифмирование скошенных признаков---------")
print("\nАсимметрия признаков ДО:")
for col in data_clean.columns:
    skew = data_clean[col].skew()
    print(f"{col}: {skew:.2f}")
data_before_log = data_clean.copy()

for col in data_clean.columns:
    skew = data_clean[col].skew()
    if abs(skew) > 1:
        # Проверяем, что все значения > 0 для логарифма
        data_clean[col] = np.log1p(data_clean[col])  # log(1+x) для защиты от 0
        print(f"Логарифмирован {col} (skew был {skew:.2f})")

print("\nАсимметрия признаков ПОСЛЕ логарифмирования:")
for col in data_clean.columns:
    skew = data_clean[col].skew()
    print(f"{col}: {skew:.2f}")

# Гистограммы ДО и ПОСЛЕ логарифмирования
# Показываем только те колонки, которые были логарифмированы
log_cols = [col for col in data_clean.columns if abs(data_before_log[col].skew()) > 1]
if log_cols:
    plt.figure(figsize=(15, 5 * len(log_cols)))
    for i, col in enumerate(log_cols, 1):
        # ДО
        plt.subplot(len(log_cols), 2, 2*i - 1)
        sns.histplot(data_before_log[col], kde=True)
        plt.title(f'ДО логарифмирования: {col}')
        plt.xlabel(col)
        # ПОСЛЕ
        plt.subplot(len(log_cols), 2, 2*i)
        sns.histplot(data_clean[col], kde=True)
        plt.title(f'ПОСЛЕ логарифмирования: {col}')
        plt.xlabel(f'log({col})')
    plt.tight_layout()
    plt.show()
else:
    print("Признаки с сильной асимметрией не найдены.")

print("\nКатегориальных переменных в датасете нет, кодирование не требуется.")
data_encoded = data_clean.copy()








print("\n----------------Подготовка к регрессии----------------")
target_col = 'SSPL'
print("\nРазделение на train/test (70/30):")
X = data_encoded.drop(columns=[target_col])  # признаки
y = data_encoded[target_col]  # целевая переменная
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42  # для воспроизводимости
)
print(f"Размер обучающей выборки (train): {X_train.shape[0]} строк")
print(f"Размер тестовой выборки (test): {X_test.shape[0]} строк")

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.histplot(y_train, kde=True)
plt.title('Train')
plt.ylabel('Частота')
plt.subplot(1, 2, 2)
sns.histplot(y_test, kde=True)
plt.title('Test')
plt.ylabel('Частота')
plt.tight_layout()
plt.show()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
print("\nМасштабирование признаков выполнено (StandardScaler)")

corr_matrix = data_encoded.corr()
sns.heatmap(corr_matrix.corr(), annot=True, fmt=".2f")
plt.title('Матрица корреляций')
plt.show()
print("\nКорреляции с SSPL:")
print(corr_matrix[target_col].sort_values(ascending=False))








# Полиномиальная регрессия 2-й степени
X_freq = data_encoded[['f']].values
y_target = data_encoded['SSPL'].values
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_freq)
model = LinearRegression()
model.fit(X_poly, y_target)
X_plot = np.linspace(X_freq.min(), X_freq.max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X_freq, y_target, alpha=0.5, s=10, label='Данные')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Полином 2-й степени')
plt.xlabel('Частота (f)')
plt.ylabel('Уровень шума (SSPL)')
plt.title('Полиномиальная регрессия 2-й степени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()