import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def linear_regression(X, y):    
    X = np.column_stack((np.ones(X.shape[0]), X))
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def predict(X, beta):
    X = np.column_stack((np.ones(X.shape[0]), X))
    return X @ beta

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) * (y_true - y_pred))
    ss_tot = np.sum((y_true - np.mean(y_true)) * (y_true - np.mean(y_true)))
    return 1 - ss_res / ss_tot



df = pd.read_csv('Student_Performance.csv') 


print(df.describe())




df.hist(bins=10, figsize=(10, 8), color='skyblue', edgecolor='black')
plt.show()




df['Hours Studied'] = df['Hours Studied'].fillna(df['Hours Studied'].mean())
df['Previous Scores'] = df['Previous Scores'].fillna(df['Previous Scores'].mean())
df['Sleep Hours'] = df['Sleep Hours'].fillna(df['Sleep Hours'].mean())



columns_to_normalize  = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced','Performance Index']

df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())


print(df.head())


df.hist(bins=10, figsize=(10, 8), color='skyblue', edgecolor='black')
plt.show()

train_df = df.sample(frac=0.8, random_state=666)
test_df = df.drop(train_df.index)


X_train = train_df.drop(columns='Performance Index')
y_train = train_df['Performance Index']
X_test = test_df.drop(columns='Performance Index')
y_test = test_df['Performance Index']

x_train1 = X_train[['Hours Studied']]
x_test1 = X_test[['Hours Studied']]


model1 = linear_regression(x_train1.values, y_train.values)


y_pred_train_model1 = predict(x_train1.values, model1)
y_pred_test_model1 = predict(x_test1.values, model1)



x_train2 = X_train[['Hours Studied', 'Sleep Hours']]
x_test2 = X_test[['Hours Studied', 'Sleep Hours']]


model2 = linear_regression(x_train2.values, y_train.values)


y_pred_train_model2 = predict(x_train2.values, model2)
y_pred_test_model2 = predict(x_test2.values, model2)




x_train3 = X_train[['Previous Scores', 'Sample Question Papers Practiced']]
X_test3 = X_test[['Previous Scores', 'Sample Question Papers Practiced']]


model3 = linear_regression(x_train3.values, y_train.values)


y_pred_train_model3 = predict(x_train3.values, model3)
y_pred_test_model3 = predict(X_test3.values, model3)





r2_model1_train = r_squared(y_train.values, y_pred_train_model1)
r2_model1_test = r_squared(y_test.values, y_pred_test_model1)


r2_model2_train = r_squared(y_train.values, y_pred_train_model2)
r2_model2_test = r_squared(y_test.values, y_pred_test_model2)


r2_model3_train = r_squared(y_train.values, y_pred_train_model3)
r2_model3_test = r_squared(y_test.values, y_pred_test_model3)


print(f"1 - R^2 (train): {r2_model1_train:.4f}, R^2 (test): {r2_model1_test:.4f}")
print(f"2 - R^2 (train): {r2_model2_train:.4f}, R^2 (test): {r2_model2_test:.4f}")
print(f"3 - R^2 (train): {r2_model3_train:.4f}, R^2 (test): {r2_model3_test:.4f}")


df['Efficiency'] = df['Hours Studied'] / df['Sleep Hours']


train_df = df.sample(frac=0.8, random_state=666)
test_df = df.drop(train_df.index)

x_train_4 = train_df[['Hours Studied', 'Previous Scores', 'Efficiency']].values
x_test_4 = test_df[['Hours Studied', 'Previous Scores', 'Efficiency']].values


model4 = linear_regression(x_train_4, y_train.values)


y_pred_train_model4 = predict(x_train_4, model4)
y_pred_test_model4 = predict(x_test_4, model4)


r2_model4_train = r_squared(y_train.values, y_pred_train_model4)
r2_model4_test = r_squared(y_test.values, y_pred_test_model4)



print(f"4 - R^2 (train): {r2_model4_train:.4f}, R^2 (test): {r2_model4_test:.4f}")

