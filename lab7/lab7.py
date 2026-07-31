import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('train.csv')
print("Первые 5 строк данных:")
print(data.head())
print(f"\nРазмер датасета: {data.shape[0]} строки и {data.shape[1]} столбца")
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
for i, col in enumerate(numeric_cols[:6], 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col], kde=True)
    plt.title(f'Распределение {col}')
    plt.ylabel('Частота')
plt.tight_layout()
plt.show()
# Boxplot
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:6], 1):
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

target_col = 'critical_temp'
features = [col for col in data_clean.columns if col != target_col]
for col in features:
    data_clean = remove_outliers_iqr(data_clean, col)
print("\nОбработка выбросов проведена методом IQR")
print(f"Размер после удаления выбросов: {data_clean.shape}")

# Боксплоты ДО и ПОСЛЕ
plt.figure(figsize=(15, 10))
for i, col in enumerate(data.columns[:6], 1):
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
# Первые 3 гистограммы ДО и ПОСЛЕ логарифмирования
data_before_log = data_clean.copy()
log_cols = [col for col in data_clean.columns if abs(data_before_log[col].skew()) > 1]
if log_cols:
    log_cols_show = log_cols[:3]
    plt.figure(figsize=(15, 5 * len(log_cols_show)))
    for i, col in enumerate(log_cols_show, 1):
        # ДО
        plt.subplot(len(log_cols_show), 2, 2*i - 1)
        sns.histplot(data_before_log[col], kde=True)
        plt.title(f'ДО логарифмирования: {col}')
        plt.xlabel(col)
        plt.ylabel('Частота')
        
        # ПОСЛЕ
        plt.subplot(len(log_cols_show), 2, 2*i)
        sns.histplot(data_clean[col], kde=True)
        plt.title(f'ПОСЛЕ логарифмирования: {col}')
        plt.xlabel(f'log({col})')
        plt.ylabel('Частота')
    plt.tight_layout()
    plt.show()
else:
    print("Признаки с сильной асимметрией не найдены.")

print("\nКатегориальных переменных в датасете нет, кодирование не требуется.")
data_encoded = data_clean.copy()





print("\n----------------Подготовка к регрессии----------------")
target_col = 'critical_temp'
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




print("\n-----------------------Регрессия----------------------")
models_reg = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}
results_reg = {}
for name, model in models_reg.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results_reg[name] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    print(f"{name:20} MSE: {results_reg[name]['MSE']:.4f}  MAE: {results_reg[name]['MAE']:.4f}  R2: {results_reg[name]['R2']:.4f}")
print("\nТаблица результатов регрессии:")
print(pd.DataFrame(results_reg).T)






print("\n------------------Классификация------------------")
data_encoded['is_high_temp'] = (data_encoded[target_col] > 50).astype(int)
print(f"Распределение классов:")
print(data_encoded['is_high_temp'].value_counts())
if len(data_encoded['is_high_temp'].unique()) < 2:
    print("В данных только один класс! Используем медиану как порог.")
    # Используем медиану как порог вместо 50
    threshold = data_encoded[target_col].median()
    data_encoded['is_high_temp'] = (data_encoded[target_col] > threshold).astype(int)
    print(f"Новый порог: {threshold:.2f}")
    print(data_encoded['is_high_temp'].value_counts())
X_clf = data_encoded.drop(columns=[target_col, 'is_high_temp'])
y_clf = data_encoded['is_high_temp']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)
models_clf = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}
results_clf = {}
for name, model in models_clf.items():
    model.fit(X_train_clf_scaled, y_train_clf)
    y_pred_clf = model.predict(X_test_clf_scaled)
    results_clf[name] = {
        'Accuracy': accuracy_score(y_test_clf, y_pred_clf),
        'Precision': precision_score(y_test_clf, y_pred_clf),
        'F1': f1_score(y_test_clf, y_pred_clf)
    }
    print(f"{name:20} Accuracy: {results_clf[name]['Accuracy']:.4f}  Precision: {results_clf[name]['Precision']:.4f}  F1: {results_clf[name]['F1']:.4f}")
print("\nТаблица результатов классификации:")
print(pd.DataFrame(results_clf).T)




# ---------------Важность признаков----------------
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_clf_scaled, y_train_clf)
feature_importance = pd.DataFrame({
    'feature': X_train_clf.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Топ-15 важнейших признаков (Random Forest)')
plt.xlabel('Важность')
plt.tight_layout()
plt.show()
