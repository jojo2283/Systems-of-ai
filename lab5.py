import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # атрибут для разделения
        self.threshold = threshold  # порог для разделения
        self.left = left  # левое поддерево
        self.right = right  # правое поддерево
        self.value = value  # значение узла (если это лист)

class DecisionTree:
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(value=y.iloc[0])

        if X.shape[1] == 0:
            return Node(value=y.mode()[0] if not y.empty else None)

        n_features = X.shape[1]
        n_random_features = int(np.sqrt(n_features))
        random_features = np.random.choice(X.columns, n_random_features, replace=False)

        best_feature, best_threshold = self._best_split(X[random_features], y)
        if best_feature is None:
            return Node(value=y.mode()[0] if not y.empty else None)

        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold
        
        # Проверка, не пустые ли множества после разбиения
        if not left_indices.any() or not right_indices.any():
            return Node(value=y.mode()[0] if not y.empty else None)

        left_node = self._build_tree(X[left_indices], y[left_indices])
        right_node = self._build_tree(X[right_indices], y[right_indices])

        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)

        # Разделение данных
        left_indices = X[feature] <= threshold
        right_indices = X[feature] > threshold
        
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0  

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        
        left_ratio = len(y[left_indices]) / len(y)
        right_ratio = len(y[right_indices]) / len(y)

        child_entropy = left_ratio * left_entropy + right_ratio * right_entropy
        gain = parent_entropy - child_entropy

        return gain

    def _entropy(self, y):
        class_counts = y.value_counts()
        probabilities = class_counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities )) 

    def predict(self, X):
        return np.array([self._predict(sample, self.root) for _, sample in X.iterrows()])

    def _predict(self, sample, node):
        if node.value is not None:
            return node.value  
        if sample[node.feature] <= node.threshold:
            return self._predict(sample, node.left)
        else:
            return self._predict(sample, node.right)
        




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

    return accuracy, precision, recall


def compute_roc_curve(y_true, y_scores):
    thresholds = np.arange(0, 1.01, 0.01)
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tpr.append(tp / (tp + fn))  # Recall
        fpr.append(fp / (fp + tn) )  

    return fpr, tpr, thresholds

def compute_pr_curve(y_true, y_scores):
    thresholds = np.arange(0, 1.01, 0.01)
    precision = []
    recall = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision.append(tp / (tp + fp) )
        recall.append(tp / (tp + fn))

    return precision, recall


# Пример использования
if __name__ == "__main__":

    df = pd.read_csv('students.csv')  

    goal = 5
    df['SUCCESS'] = df['GRADE'].apply(lambda x: 1 if x >= goal else 0)

    # Извлекаем оценки и метки
    X = df.drop(columns=['STUDENT ID', 'COURSE ID', 'GRADE', 'SUCCESS'])
    y = df['SUCCESS']

    # Обучение дерева решений
    tree = DecisionTree()
    tree.fit(X, y)

    # Ручное разбиение на обучающую и тестовую выборки
    np.random.seed(668)  # Для воспроизводимости
    train_size = int(0.8 * len(df))
    indices = np.random.permutation(len(df))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    
    tree = DecisionTree()
    tree.fit(X_train, y_train)

    
    predictions = tree.predict(X_test)

    
    accuracy, precision, recall = calculate_metrics(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    
    # Вычисление ROC и PR кривых
    fpr, tpr, thresholds_roc = compute_roc_curve(y_test, predictions)
    precision2, recall2 = compute_pr_curve(y_test, predictions)
   
    # Построение ROC-кривой
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid()

    # Построение PR-кривой
    plt.subplot(1, 2, 2)
    plt.plot(recall2, precision2, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()

    plt.tight_layout()
    plt.show()


