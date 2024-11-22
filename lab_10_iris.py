import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import math

# Clasificador Euclidiano
class EuclideanClassifier:
    def __init__(self, k=1):
        self.X_train = []
        self.y_train = []
        self.k = k  # Soporte para K-vecinos
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            closest_classes = self._find_closest(test_point)
            # Predicción basada en mayoría (para k>1)
            predicted_class = max(set(closest_classes), key=closest_classes.count)
            predictions.append(predicted_class)
        return predictions
    
    def _find_closest(self, test_point):
        distances = []
        for i, train_point in enumerate(self.X_train):
            dist = self._euclidean_distance(test_point, train_point)
            distances.append((dist, self.y_train[i]))
        # Ordenar por distancia y tomar los k vecinos más cercanos
        distances.sort(key=lambda x: x[0])
        return [neighbor[1] for neighbor in distances[:self.k]]
    
    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# Función para cargar el dataset de Iris
def load_iris_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', dtype=str)
    X = data[:, :-1].astype(float)  # Características
    y = data[:, -1]                # Etiquetas
    return X, y

# Métodos de validación
def hold_out(X, y, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, confusion

def k_fold_cross_validation(X, y, classifier, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    confusions = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        confusions.append(confusion_matrix(y_test, y_pred))
    return np.mean(accuracies), confusions

def leave_one_out_cross_validation(X, y, classifier):
    loo = LeaveOneOut()
    accuracies = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return np.mean(accuracies)

# Función principal para ejecutar el análisis
def main():
    # Ruta fija al archivo del dataset Iris
    file_path = '/content/iris.data'
    
    # Cargar datos
    X, y = load_iris_data(file_path)
    
    # Convertir etiquetas a números (si es necesario)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Clasificador Euclidiano con k=3
    classifier = EuclideanClassifier(k=3)
    
    # Hold-Out Validation
    accuracy_hold_out, confusion_hold_out = hold_out(X, y_encoded, classifier)
    print("\nHold-Out Validation:")
    print(f"Accuracy: {accuracy_hold_out:.2f}")
    print(f"Confusion Matrix:\n{confusion_hold_out}\n")
    
    # 10-Fold Cross-Validation
    accuracy_k_fold, confusions_k_fold = k_fold_cross_validation(X, y_encoded, classifier, k=10)
    print("10-Fold Cross-Validation:")
    print(f"Mean Accuracy: {accuracy_k_fold:.2f}\n")
    
    # Leave-One-Out Cross-Validation
    accuracy_loo = leave_one_out_cross_validation(X, y_encoded, classifier)
    print("Leave-One-Out Cross-Validation:")
    print(f"Mean Accuracy: {accuracy_loo:.2f}\n")

# Ejecutar el análisis
if __name__ == "__main__":
    main()
