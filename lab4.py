import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Функции для k-NN
def euclidean_distance(train_data, test_row):
    return np.sqrt(np.sum((train_data - test_row) ** 2, axis=1))

def knn_classify(train_data, train_labels, test_row, k):
    distances = euclidean_distance(train_data, test_row)
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = train_labels.iloc[k_indices]
    prediction = k_nearest_labels.mode()[0]
    return prediction

def evaluate_knn(train_data, train_labels, test_data, test_labels, k):
    predictions = []
    for _, test_row in test_data.iterrows():
        prediction = knn_classify(train_data, train_labels, test_row, k)
        predictions.append(prediction)
    cm = confusion_matrix_manual(test_labels, predictions)
    return cm

def confusion_matrix_manual(actual, predicted):
    unique_classes = np.unique(actual)
    matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    for i in range(len(actual)):
        actual_class = actual.iloc[i] 
        predicted_class = predicted[i]
        matrix[int(actual_class)][int(predicted_class)] += 1
    return matrix

# Загрузка данных
df = pd.read_csv('diabetes.csv')
#print(df.describe())

# Заполнение пропусков
for i in df:
    if i != 'Outcome':
        df[i] = df[i].fillna(df[i].mean())

# Нормализация
columns_to_normalize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X.boxplot(patch_artist=True)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Glucose'], df['Insulin'], df['BMI'], c=df['Outcome'], cmap='coolwarm')

# Set labels for axes
ax.set_xlabel('Glucose')
ax.set_ylabel('Insulin')
ax.set_zlabel('BMI')


plt .show()



# Разделение данных на обучающие и тестовые наборы
train_data = X.iloc[:int(0.8 * len(df))]
train_labels = y.iloc[:int(0.8 * len(df))]
test_data = X.iloc[int(0.8 * len(df)):]
test_labels = y.iloc[int(0.8 * len(df)):]

# Функция для случайного выбора признаков
def random_features(X, num_features):
    return X.sample(n=num_features, axis=1, random_state=42)

# Фиксированные признаки
fixed_features = ['Glucose', 'Insulin', 'BMI']
X_fixed = X[fixed_features]
train_data_fixed = X_fixed.iloc[:int(0.8 * len(df))]
test_data_fixed = X_fixed.iloc[int(0.8 * len(df)):]

# Случайные признаки
random_feature_count = 3
X_random = random_features(X, random_feature_count)
train_data_random = X_random.iloc[:int(0.8 * len(df))]
test_data_random = X_random.iloc[int(0.8 * len(df)):]

# Оценка при разных значениях k
k_values = [3, 5, 10]

# Модель с фиксированными признаками
print("Fixed Feature Model Evaluation:")
for k in k_values:
    cm_fixed = evaluate_knn(train_data_fixed, train_labels, test_data_fixed, test_labels, k)
    print(f'Confusion Matrix with k={k} (Fixed features):')
    print(cm_fixed)

# Модель со случайными признаками
print("Random Feature Model Evaluation:")
for k in k_values:
    cm_random = evaluate_knn(train_data_random, train_labels, test_data_random, test_labels, k)
    print(f'Confusion Matrix with k={k} (Random features):')
    print(cm_random)
