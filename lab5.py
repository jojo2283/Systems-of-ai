import numpy as np
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Node:
  def __init__(self, parent_attribute = None, parent_attribute_value = None, attribute = None, entropy = 0.0, samples_count = 0):
    self.parent_attribute = parent_attribute
    self.parent_attribute_value = parent_attribute_value
    self.attribute = attribute
    self.entropy = entropy
    self.samples_count = samples_count
    self.samples = dict()
    self.probability = dict()
    self.prediction = None
    self.children = list()

  def predict(self, X):
    for child in self.children:
      if X[self.attribute] == child.parent_attribute_value:
        return child.predict(X)

    return self.prediction


  def predict_proba(self, X):
    for child in self.children:
      if X[self.attribute] == child.parent_attribute_value:
        return child.predict_proba(X)

    return self.probability


class Calc:
  def __init__(self, df: pd.DataFrame, y_label: int):
    self.y_label = y_label
    self.y_classes = set(df[y_label].to_list())
    self.X_values = dict()
    for label in df.columns:
      if label != y_label:
        self.X_values[label] = set(df[label].to_list())

  def freq(self, df, C_j):
    return df.loc[df[self.y_label] == C_j].shape[0]
  def info(self, df):
    if df.shape[0] == 0:
      return 0

    result = 0
    for y_class in self.y_classes:
      freq_c_div_df = self.freq(df, y_class) / df.shape[0]
      if freq_c_div_df == 0:
        continue
      result -= freq_c_div_df * np.log2(freq_c_div_df)
    return result

  def info_X(self, df, X_label):
    if df.shape[0] == 0:
      return 0

    result = 0
    for attr in self.X_values[X_label]:
      df_i = df.loc[df[X_label] == attr]
      if df_i.shape[0] == 0:
        continue
      result += df_i.shape[0] * self.info(df_i)
    result /= df.shape[0]
    return result

  def split_info_X(self, df, X_label):
    result = 1e-9
    for attr in self.X_values[X_label]:
      df_i = df.loc[df[X_label] == attr]
      if df_i.shape[0] == 0:
        continue
      df_i_div_df = df_i.shape[0] / df.shape[0]
      result -= df_i_div_df * np.log2(df_i_div_df)
    return result

  def gain_ratio_X(self, df, X_label):
    return (self.info(df) - self.info_X(df, X_label)) / self.split_info_X(df, X_label)



class DecisionTree:
  def __init__(self, max_leaf_entropy , max_leaf_samples):
    

    self.decision_tree_node = None
    self.max_leaf_entropy = max_leaf_entropy
    self.max_leaf_samples = max_leaf_samples
    self.info_entropy = None

  def build_tree(self, df: pd.DataFrame, TreeNode):

    if df.shape[0] == 0:
      return

    best_attr = None
    best_ratio = 0
    for attr in self.info_entropy.X_values:
      ratio = self.info_entropy.gain_ratio_X(df, attr)
      
      if best_ratio < ratio:
        best_attr = attr
        best_ratio = ratio
   

    TreeNode.attribute = best_attr
    TreeNode.entropy = best_ratio
    max_samples_count = 0

    for y_class in self.info_entropy.y_classes:
      TreeNode.samples[y_class] = df.loc[df[self.info_entropy.y_label] == y_class].shape[0]
      TreeNode.probability[y_class] = TreeNode.samples[y_class] / df.shape[0]
      if max_samples_count < TreeNode.samples[y_class]:
        max_samples_count = TreeNode.samples[y_class]
        TreeNode.prediction = y_class

    TreeNode.samples_count = df.shape[0]

    if (TreeNode.entropy > self.max_leaf_entropy) and (TreeNode.samples_count > self.max_leaf_samples):
      for attr in self.info_entropy.X_values[best_attr]:
        df_loc = df.loc[df[best_attr] == attr]
        
        if df_loc.shape[0] > 0:
          child = Node()
          child.parent_attribute = best_attr
          child.parent_attribute_value = attr
          TreeNode.children.append(child)
          self.build_tree(df_loc, TreeNode.children[-1])

  def fit(self, df, y_label):
    self.info_entropy = Calc(df, y_label)
    self.decision_tree_node = Node()
    self.build_tree(df, self.decision_tree_node)
    return self

  def predict(self, X_test):
    y_test = []
    for i in range(X_test.shape[0]):
      y_test.append(self.decision_tree_node.predict(X_test.iloc[i]))
    return y_test

  def predict_proba(self, X_test):
    y_test = []
    for i in range(X_test.shape[0]):
      y_test.append(self.decision_tree_node.predict_proba(X_test.iloc[i]))
    return y_test



df = pd.read_csv('students.csv')
goal = 5
df['SUCCESS'] = df['GRADE'].apply(lambda x: 1 if x >= goal else 0)

X_labels_count = int(np.round(np.sqrt(len(df.columns)-4)))+1
X_labels = df.columns[1:].to_list()
X_labels.remove('COURSE ID')
X_labels.remove('GRADE')
X_labels.remove('SUCCESS')
shuffle(X_labels) 
X_labels = X_labels[:X_labels_count]

X_attributes = dict()
for label in X_labels:
  X_i_set = set(df[label].to_list())
  X_attributes[label] = X_i_set

y_label = "SUCCESS"
y_classes = set(df[y_label].to_list())
y_classes_count=2

data_n = df[X_labels + [y_label]]

X_train, X_test, y_train, y_test = train_test_split(data_n.drop(columns=[y_label]), data_n[y_label], test_size=0.2, random_state=0)

df_train = X_train.join(y_train)
y_test = y_test.to_list()

dt = DecisionTree(0.001, 10).fit(df_train, y_label)
predictions = dt.predict(X_test)

def confusion(y_true, y_pred):
  TP, FP, FN, TN = 0, 0, 0, 0
  for i in range(len(y_true)):
    if y_pred[i] == 1:
      if y_true[i] == 1:
        TP += 1
      else:
        FP += 1
    else:
      if y_true[i] == 1:
        FN += 1
      else:
        TN += 1
  return TP, FP, FN, TN



def TPR_by_FPR(y_true, y_probs, y_positive = 1, y_negative = 0): 
  y_probs_sorted = sorted([v[y_positive] for v in y_probs], reverse = True)
  points_count = len(y_probs_sorted)
  
  FPR_values, TPR_values = [0], [0]
  last_line_value = -1000
  for i in range(points_count):
    classification_line_value = y_probs_sorted[i]
    if abs(last_line_value - classification_line_value) < 1e-3:
      continue
    last_line_value = classification_line_value
    
    y_pred = []

    for j in range(len(y_probs)):
      y_pred.append(y_positive if y_probs[j][y_positive] >= classification_line_value else y_negative)
    TP, FP, FN, TN = confusion(y_true, y_pred)

    try:
      FPR = FP / (TN + FP)
      TPR = TP / (TP + FN)

      FPR_values.append(FPR)
      TPR_values.append(TPR)
    except:
      pass

  return FPR_values, TPR_values

def Precision_by_Recall(y_true, y_probs, y_positive , y_negative):
  y_probs_sorted = sorted([v[y_positive] for v in y_probs], reverse = True)
  points_count = len(y_probs_sorted)

  Recall_values, Precision_values = [0], [1]
  last_line_value = -1000
  for i in range(points_count):
    classification_line_value = y_probs_sorted[i]
    if abs(last_line_value - classification_line_value) < 1e-3:
      continue
    last_line_value = classification_line_value

    y_pred = []
    for j in range(len(y_probs)):
      y_pred.append(y_positive if y_probs[j][y_positive] >= classification_line_value else y_negative)
    TP, FP, FN, TN = confusion(y_true, y_pred)

    try:
      Recall = TP / (TP + FN)
      Precision = TP / (TP + FP)

      Recall_values.append(Recall)
      Precision_values.append(Precision)
    except:
      pass

  return Recall_values, Precision_values

probs = dt.predict_proba(X_test)



TP, FP, FN, TN = confusion(y_test, predictions)
print("TP, FP, FN, TN = {}, {}, {}, {}".format(TP, FP, FN, TN))
print("Precision = {}".format(TP / (TP + FP)))
print("Recall = {}".format(TP / (TP + FN)))
print("Accuracy = {}".format((TP + TN) / (TP + FP + FN + TN)))
print(X_test.columns.to_list())

roc_x, roc_y = TPR_by_FPR(y_test, probs, 1, 0) 
pr_x, pr_y = Precision_by_Recall(y_test, probs, 1, 0)


# roc
plt.plot(roc_x, roc_y, 's-', markersize = 4, label = 'Receiver Operating Characteristic')

# y=x line
plt.plot([0, 1], [0, 1], '--')
plt.xlim(-0.1, 1.3)
plt.ylim(-0.1, 1.3)
plt.legend(loc='upper right')

plt.show()


plt.plot(pr_x, pr_y, 'o-', markersize = 4, label = 'Precision Recall')
plt.xlim(-0.1, 1.3)
plt.ylim(-0.1, 1.3)

plt.show()

def integrate_traps(x_values, y_values):
  return sum([(y_values[i] + y_values[i + 1]) * (x_values[i + 1] - x_values[i]) / 2 for i in range(len(x_values) - 1)])
     

print("Area under curve ROC = {}".format(integrate_traps(roc_x, roc_y)))
print("Area under curve PR = {}".format(integrate_traps(pr_x, pr_y)))
     