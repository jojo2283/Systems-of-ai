import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def euclidean_distance(train_data, test_row):
    # Векторизованный расчет евклидова расстояния
    return np.sqrt(np.sum((train_data - test_row) ** 2, axis=1))

def knn_classify(train_data, train_labels, test_row, k):
    # Рассчитываем расстояния сразу для всех строк
    distances = euclidean_distance(train_data, test_row)
    
    # Получаем индексы k ближайших соседей
    k_indices = np.argsort(distances)[:k]
    
    # Берем соответствующие им метки классов
    k_nearest_labels = train_labels.iloc[k_indices]
    
    # Возвращаем наиболее часто встречающийся класс
    prediction = k_nearest_labels.mode()[0]
    
    return prediction

def evaluate_knn(train_data, train_labels, test_data, test_labels, k):
    predictions = []
    
    for _, test_row in test_data.iterrows():
        prediction = knn_classify(train_data, train_labels, test_row, k)
        predictions.append(prediction)
    
    # Построение матрицы ошибок вручную
    cm = confusion_matrix_manual(test_labels, predictions)
    return cm


def random_features_knn(train_data, test_data, k, num_features):
    features = np.random.choice(train_data.columns[:-1], num_features, replace=False)
    return evaluate_knn(train_data[features + ['Outcome']], test_data[features + ['Outcome']], k)

def fixed_features_knn(train_data, test_data, k, selected_features):
    return evaluate_knn(train_data[selected_features + ['Outcome']], test_data[selected_features + ['Outcome']], k)

def confusion_matrix_manual(actual, predicted):
    unique_classes = np.unique(actual)
    matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    
    for i in range(len(actual)):
        actual_class = actual.iloc[i]  # Используем iloc для доступа к элементу по позиции
        predicted_class = predicted[i]  # Здесь predicted - список, его можно индексировать напрямую
        matrix[int(actual_class)][int(predicted_class)] += 1
    
    return matrix



df = pd.read_csv('diabetes.csv') 
#print(df.describe())

# df.hist(bins=10, figsize=(10, 8), color='skyblue', edgecolor='black')
# plt.show()


for i in df:
    if i!='Outcome':
        df[i] = df[i].fillna(df[i].mean())

columns_to_normalize  = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI', 'Pedigree','Age']

df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X.boxplot(patch_artist=True)
# plt.show()
print(df.head())


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Glucose'], df['Insulin'], df['BMI'], c=df['Outcome'], cmap='coolwarm')

# Set labels for axes
ax.set_xlabel('Glucose')
ax.set_ylabel('Insulin')
ax.set_zlabel('BMI')


plt.show()



train_data = X.iloc[:int(0.8 * len(df))]
train_labels = y.iloc[:int(0.8 * len(df))]
test_data = X.iloc[int(0.8 * len(df)):]
test_labels = y.iloc[int(0.8 * len(df)):]


selected_features = ['Glucose', 'BMI', 'Age']
k_values = [3, 5, 10]

for k in k_values:
    cm_fixed = evaluate_knn(train_data, train_labels, test_data, test_labels, k)
    print(f'Confusion Matrix with k={k}:')
    print(cm_fixed)
