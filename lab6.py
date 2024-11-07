import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, method='gradient_descent'):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.method = method
        self.weights = None
        self.bias = None

   
    def fit(self, X, y):

        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if self.method == 'gradient_descent':
            for _ in range(self.iterations):
                linear_model = np.dot(X, self.weights) + self.bias
                y_pred = sigmoid(linear_model)

                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        elif self.method == 'newton':
            for _ in range(self.iterations):
                linear_model = np.dot(X, self.weights) + self.bias
                y_pred = sigmoid(linear_model)

        
                gradient = np.dot(X.T, (y_pred - y))

        
                diagonal = y_pred * (1 - y_pred)
                hessian = np.dot(X.T * diagonal, X)

        
                reg_lambda = 1e-4  
                hessian_reg = hessian + reg_lambda * np.eye(X.shape[1])

        
                self.weights -= np.linalg.solve(hessian_reg, gradient)
                self.bias -= np.sum(gradient)

    
    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
    
   
    def predict(self, X, threshold=0.5):
        y_pred_prob = self.predict_prob(X)
        return [1 if i > threshold else 0 for i in y_pred_prob]

def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)


def calculate_metrics(y_true, y_pred):
    tp = fp = fn = tn= 0
    
    for true, pred in zip(y_true, y_pred):
        if pred == 1 and true == 1:
            tp += 1 
        elif pred == 1 and true == 0:
            fp += 1  
        elif pred == 0 and true == 1:
            fn += 1 
        else:
            tn+=1


    accuracy = (tp + tn) / len(y_true) 
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn) 
    f1 = 2 * (precision * recall) / (precision + recall)


    return f1, accuracy, precision, recall

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positive = np.sum(y_pred == 1)
    if predicted_positive == 0:    
        return 0
    return tp / predicted_positive

def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    actual_positive = np.sum(y_true == 1)
    if actual_positive == 0:
        return 0
    return true_positive / actual_positive

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data.values) 
    
    f1, accuracy, precision, recall = calculate_metrics(test_labels,predictions)
    logloss = log_loss(test_labels,predictions)
    return logloss, accuracy, precision, recall, f1

df = pd.read_csv('diabetes.csv')
#print(df.describe())


for i in df:
    if i != 'Outcome':
        df[i] = df[i].fillna(df[i].mean())


columns_to_normalize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X.boxplot(patch_artist=True)
plt.show()


train_data = X.iloc[:int(0.8 * len(df))]
train_labels = y.iloc[:int(0.8 * len(df))]
test_data = X.iloc[int(0.8 * len(df)):]
test_labels = y.iloc[int(0.8 * len(df)):]


model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(train_data.values, train_labels.values)




learning_rates = [0.01, 0.01, 0.2]
iterations_list = [500, 1000, 2000]
methods = ['gradient_descent', 'newton']

# Результаты экспериментов
results = []

# Перебор комбинаций гиперпараметров
for lr in learning_rates:
    for iterations in iterations_list:
        for method in methods:
            
            
            # Создание и обучение модели
            model = LogisticRegression(learning_rate=lr, iterations=iterations, method=method)
            model.fit(train_data.values, train_labels.values)
            
            # Оценка модели с logloss
            logloss, accuracy, precision, recall, f1 = evaluate_model(model, test_data, test_labels)

            # Запись результатов
            results.append({
                'learning_rate': lr,
                'iterations': iterations,
                'method': method,
                'log_loss': logloss,  
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            

# Вывод результатов
results_df = pd.DataFrame(results)
print(results_df)