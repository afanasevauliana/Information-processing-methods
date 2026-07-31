import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

data = pd.read_csv('train.csv')
print("Первые 5 строк данных:")
print(data.head())
print(f"\nРазмер датасета: {data.shape[0]} строки и {data.shape[1]} столбцов")
print("\nТипы данных:")
print(data.dtypes)


target_col = 'SSPL'
features = ['f', 'alpha', 'c', 'U_infinity', 'delta']
print(f"\nЦелевая переменная: {target_col}")

plt.figure(figsize=(10, 6))
plt.scatter(data['f'], data['SSPL'], alpha=0.5, s=15)
plt.xlabel('Частота (f)')
plt.ylabel('SSPL')
plt.title('Зависимость SSPL от частоты')
plt.grid(True, alpha=0.3)
plt.show()

# --------------------Ядерная регрессия--------------------
X = data[features]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
kr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
kr.fit(X_train_scaled, y_train)
y_pred_kr = kr.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_test['f'], y_test, alpha=0.5, s=20, label='Фактические значения')

sort_idx = X_test['f'].argsort()
plt.plot(X_test['f'].iloc[sort_idx], y_pred_kr[sort_idx], 
         'r-', linewidth=2, label='Ядерная регрессия')
plt.xlabel('Частота (f)')
plt.ylabel('SSPL')
plt.title('Ядерная регрессия')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()





# --------------------K-NN регрессия--------------------
plt.figure(figsize=(12, 7))
plt.scatter(X_test['f'], y_test, alpha=0.3, s=15, color='gray', label='Фактические значения')
k_values = [3, 5, 10]
colors = ['green', 'blue', 'orange']
for k, color in zip(k_values, colors):
    # Обучаем модель
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    sort_idx = X_test['f'].argsort()
    plt.plot(X_test['f'].iloc[sort_idx], y_pred_knn[sort_idx], 
             color=color, linewidth=2, label=f'k-NN (k={k})')

plt.xlabel('Частота (f)')
plt.ylabel('SSPL')
plt.title('k-NN регрессия при разных k (3, 5, 10)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()






print("\n----------------Переход к классификации---------------")
# Создание бинарного признака
y_binary = (data['SSPL'] > data['SSPL'].median()).astype(int)
data['SSPL_binary'] = y_binary
print(f"Порог (медиана): {data['SSPL'].median():.2f}")
print(f"Класс 0: {sum(y_binary == 0)}")
print(f"Класс 1: {sum(y_binary == 1)}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
counts = data['SSPL_binary'].value_counts()
plt.bar(['Класс 0', 'Класс 1'], counts.values, color=['blue', 'red'], alpha=0.7)
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.title('Гистограмма распределения бинарного признака')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
class_0 = data[data['SSPL_binary'] == 0]['SSPL']
class_1 = data[data['SSPL_binary'] == 1]['SSPL']
plt.boxplot([class_0, class_1], tick_labels=['Класс 0', 'Класс 1'])
plt.ylabel('SSPL')
plt.title('Boxplot SSPL по классам')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()





# Логисчтическая регрессия и ROC-кривая
X_bin = data[features]
y_bin = data['SSPL_binary']
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_bin, test_size=0.3, random_state=42
)
scaler_bin = StandardScaler()
X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
X_test_bin_scaled = scaler_bin.transform(X_test_bin)
logreg = LogisticRegression()
logreg.fit(X_train_bin_scaled, y_train_bin)
y_prob = logreg.predict_proba(X_test_bin_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Случайно')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()




print("\n-------------------Сравнение методов------------------")

print("\nПроверка мультиколлинеарности:")
corr_matrix = data[features].corr()
print("\nКорреляционная матрица:")
print(corr_matrix)

high_corr = False
correlated_pairs = []

for i in range(len(features)):
    for j in range(i+1, len(features)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            print(f"Сильная корреляция между {features[i]} и {features[j]}: {corr:.2f}")
            correlated_pairs.append((features[i], features[j]))
            high_corr = True

if not high_corr:
    print("Сильной корреляции нет")
    features_clean = features
else:
    # Смотрим корреляцию с целевой переменной
    corr_with_target = data[features + ['SSPL']].corr()['SSPL'].sort_values(ascending=False)
    print("\nКорреляция признаков с SSPL:")
    print(corr_with_target)
    
    # Удаляем один из коррелирующих признаков (с меньшей корреляцией с SSPL)
    drop_features = []
    for pair in correlated_pairs:
        # Определяем корреляцию каждого признака с SSPL
        corr1 = abs(corr_with_target[pair[0]])
        corr2 = abs(corr_with_target[pair[1]])
        if corr1 < corr2:
            drop_features.append(pair[0])
            print(f"\nУдаляем '{pair[0]}' (корреляция с SSPL: {corr1:.3f})")
            print(f"Оставляем '{pair[1]}' (корреляция с SSPL: {corr2:.3f})")
        else:
            drop_features.append(pair[1])
            print(f"\nУдаляем '{pair[1]}' (корреляция с SSPL: {corr2:.3f})")
            print(f"Оставляем '{pair[0]}' (корреляция с SSPL: {corr1:.3f})")
    
    features_clean = [f for f in features if f not in drop_features]
    print(f"\nИспользуем признаки: {features_clean}")


print("\n--------Зависимость accuracy от числа соседей---------")
k_range = range(1, 31)
accuracies = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_bin_scaled, y_train_bin)
    accuracies.append(accuracy_score(y_test_bin, knn.predict(X_test_bin_scaled)))
best_k = k_range[np.argmax(accuracies)]
best_acc = max(accuracies)

plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, 'bo-', linewidth=2, markersize=8)
plt.axvline(best_k, color='red', linestyle='--', label=f'Лучшее k={best_k} (acc={best_acc:.3f})')
plt.xlabel('Число соседей (k)')
plt.ylabel('Accuracy')
plt.title('Зависимость accuracy от числа соседей в k-NN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Лучшее k: {best_k}, Accuracy: {best_acc:.3f}")


print("\n--------------Сравнение accuracy моделей--------------")
logreg = LogisticRegression()
logreg.fit(X_train_bin_scaled, y_train_bin)
logreg_acc = accuracy_score(y_test_bin, logreg.predict(X_test_bin_scaled))
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_bin_scaled, y_train_bin)
knn_acc = accuracy_score(y_test_bin, knn_best.predict(X_test_bin_scaled))
kr_clf = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
kr_clf.fit(X_train_bin_scaled, y_train_bin)
kr_pred = kr_clf.predict(X_test_bin_scaled)
kr_acc = accuracy_score(y_test_bin, (kr_pred > 0.5).astype(int))
models = ['Логистическая\nрегрессия', 'k-NN\n(k='+str(best_k)+')', 'Ядерная\nрегрессия']
scores = [logreg_acc, knn_acc, kr_acc]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, scores, color=['blue', 'green', 'orange'], alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Сравнение точности моделей классификации')
plt.ylim(0, 1)
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

print(f"Логистическая регрессия: {logreg_acc:.3f}")
print(f"k-NN (k={best_k}): {knn_acc:.3f}")
print(f"Ядерная регрессия: {kr_acc:.3f}")


print("\n-----------Scatter plot значимых признаков------------")
weights = pd.DataFrame({
    'Признак': features,
    'Вес': logreg.coef_[0]
}).sort_values('Вес', ascending=False)
print("Наиболее значимые признаки (по весам логистической регрессии):")
print(weights)
top_features = weights['Признак'].head(2).tolist()

plt.figure(figsize=(10, 8))
plt.scatter(data[top_features[0]], data[top_features[1]], 
            c=data['SSPL_binary'], cmap='coolwarm', alpha=0.6, s=30)
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title(f'Scatter plot: {top_features[0]} vs {top_features[1]}\n(раскраска по классам)')
plt.colorbar(label='Класс')
plt.grid(True, alpha=0.3)
plt.show()

