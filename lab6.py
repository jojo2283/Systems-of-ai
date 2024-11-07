import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient_descent(X_train, Y_train, iterations, learning_rate):
    objects_num, characteristics_num = X_train.shape
    weights = np.zeros(characteristics_num)
    losses = []
    bias = 0
    
    for _ in range(iterations):

        t = np.dot(X_train, weights) + bias
      
        z = sigmoid(t)
        
        dw = (1 / objects_num) * np.dot(X_train.T, (z - Y_train))
        
        db = (1 / objects_num) * np.sum(z - Y_train)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db


        
        loss = log_loss(Y_train, z)
        losses.append(loss)
   
    coeff = {'weights': weights, 'bias': bias}
    return coeff, losses
     

def newton_optimization(X_train, Y_train, iterations):
    objects_num, characteristics_num = X_train.shape

    weights = np.zeros(characteristics_num)
    losses = []
    bias = 0

    for _ in range( iterations):

        t = np.dot(X_train, weights) + bias
      
        z = sigmoid(t)

        
        dw = (1 / objects_num) * np.dot(X_train.T, (z - Y_train))
        
        db = (1 / objects_num) * np.sum(z - Y_train)
        
        hessian = (1 / objects_num) * (X_train.T @ ((z * (1 - z)) * X_train.T).T)

        weights -= np.linalg.inv(hessian) @ dw
        bias -= db

        
        loss = log_loss(Y_train, z)
        losses.append(loss)
            
    coeff = {'weights': weights, 'bias': bias}
    return coeff, losses




def predict(X_test, coeff):
   
    weights = coeff['weights']
    bias = coeff['bias']
    
    t = np.dot(X_test, weights) + bias

    z = sigmoid(t)

    return np.where(z > 0.5, 1, 0)


def calculate_metrics(Y_prediction, Y_test):
    TP = np.sum((Y_prediction == 1) & (Y_test == 1))
    TN = np.sum((Y_prediction == 0) & (Y_test == 0))
    FP = np.sum((Y_prediction == 1) & (Y_test == 0))
    FN = np.sum((Y_prediction == 0) & (Y_test == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0    
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return {'accuracy': accuracy, 'precision': precision,  'recall': recall, 'f1_score': f1_score}


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

max_f1_score = 0

learning_rates = [0.05, 0.2, 0.5]
iterations_list = [500, 1000, 2000]


results = []


for lr in learning_rates:
    for iterations in iterations_list:

        coeff, losses = gradient_descent(train_data, train_labels, iterations, lr)


        Y_prediction = predict(test_data, coeff)

        metrics = calculate_metrics(Y_prediction, test_labels)
           
        results.append({
                'learning_rate': lr,
                'iterations': iterations,
                'method': "gradient_descent",
                'log_loss': losses[0] - losses[len(losses) - 1],  
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })


for iterations in iterations_list:

    coeff, losses = newton_optimization(train_data, train_labels, iterations)

    Y_prediction = predict(test_data, coeff)

    metrics = calculate_metrics(Y_prediction, test_labels)
           
    results.append({
                'learning_rate': lr,
                'iterations': iterations,
                'method': "newton_optimization",
                'log_loss': losses[0] - losses[len(losses) - 1],  
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })


results_df = pd.DataFrame(results)
print(results_df)