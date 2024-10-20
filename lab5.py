import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('students.csv')

goal = 5
df['SUCCESS'] = df['GRADE'].apply(lambda x: 1 if x >= goal else 0)

n = 30
num_features = int(np.sqrt(n))

random_features = np.random.choice(range(1, n+1), num_features, replace=False)
print(random_features)