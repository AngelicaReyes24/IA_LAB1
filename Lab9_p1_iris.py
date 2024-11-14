import math

# Clasificador Euclidiano
class EuclideanClassifier:
    def __init__(self):
        self.X_train = []
        self.y_train = []
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            closest_class = self._find_closest(test_point)
            predictions.append(closest_class)
        return predictions
    
    def _find_closest(self, test_point):
        min_distance = float('inf')
        closest_class = None
        for i, train_point in enumerate(self.X_train):
            dist = self._euclidean_distance(test_point, train_point)
            if dist < min_distance:
                min_distance = dist
                closest_class = self.y_train[i]
        return closest_class
    
    def _euclidean_distance(self, point1, point2):
        sum_squared = sum((a - b) ** 2 for a, b in zip(point1, point2))
        return math.sqrt(sum_squared)

# Métodos de Validación
def hold_out(X, y, classifier):
    split_index = int(0.7 * len(X))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = calculate_accuracy(y_test, predictions)
    confusion = calculate_confusion_matrix(y_test, predictions)
    
    return accuracy, confusion

def k_fold_cross_validation(X, y, classifier, k=10):
    fold_size = len(X) // k
    accuracies = []
    all_confusions = []
    
    for i in range(k):
        X_train = X[:i*fold_size] + X[(i+1)*fold_size:]
        y_train = y[:i*fold_size] + y[(i+1)*fold_size:]
        X_test = X[i*fold_size:(i+1)*fold_size]
        y_test = y[i*fold_size:(i+1)*fold_size]
        
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        
        accuracy = calculate_accuracy(y_test, predictions)
        confusion = calculate_confusion_matrix(y_test, predictions)
        
        accuracies.append(accuracy)
        all_confusions.append(confusion)
    
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy, all_confusions

def leave_one_out_cross_validation(X, y, classifier):
    accuracies = []
    all_confusions = []
    
    for i in range(len(X)):
        X_train = X[:i] + X[i+1:]
        y_train = y[:i] + y[i+1:]
        X_test = [X[i]]
        y_test = [y[i]]
        
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        
        accuracy = calculate_accuracy(y_test, predictions)
        confusion = calculate_confusion_matrix(y_test, predictions)
        
        accuracies.append(accuracy)
        all_confusions.append(confusion)
    
    average_accuracy = sum(accuracies) / len(accuracies)
    return average_accuracy, all_confusions

# Funciones para Calcular el Desempeño
def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Función modificada para calcular la matriz de confusión
def calculate_confusion_matrix(y_true, y_pred):
    unique_classes = set(y_true)
    class_to_index = {label: idx for idx, label in enumerate(unique_classes)}
    matrix_size = len(unique_classes)
    confusion_matrix = [[0]*matrix_size for _ in range(matrix_size)]
    
    for true, pred in zip(y_true, y_pred):
        if pred not in class_to_index:
            continue  # Ignorar predicciones desconocidas o erróneas
        i = class_to_index[true]
        j = class_to_index[pred]
        confusion_matrix[i][j] += 1
    
    return confusion_matrix

# Cargar y Preparar el Dataset Iris con codificación alternativa
def load_iris_data(file_path):
    X = []
    y = []
    
    # Leer el archivo de datos con codificación ISO-8859-1
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            line = line.strip()
            if line:  # Ignorar líneas vacías
                parts = line.split(',')
                X.append([float(value) for value in parts[:4]])
                y.append(parts[4])
    
    return X, y

# Ruta del archivo Iris
iris_file_path = '/content/iris.data'
X_iris, y_iris = load_iris_data(iris_file_path)

# Función Principal para Ejecutar el Clasificador con Métodos de Validación
def main():
    classifier = EuclideanClassifier()
    
    # Hold-Out Validation con Iris Dataset
    accuracy_hold_out, confusion_hold_out = hold_out(X_iris, y_iris, classifier)
    print("Hold-Out Validacion y Precision:", accuracy_hold_out)
    print("Hold-Out Matriz de confusion:", confusion_hold_out)
    
    # 10-Fold Cross-Validation con Iris Dataset
    accuracy_k_fold, confusions_k_fold = k_fold_cross_validation(X_iris, y_iris, classifier, k=10)
    print("\n10-Fold Cross-Validation Accuracy:", accuracy_k_fold)
    
    # Leave-One-Out Cross-Validation con Iris Dataset
    accuracy_loo, confusions_loo = leave_one_out_cross_validation(X_iris, y_iris, classifier)
    print("\nLeave-One-Out Cross-Validation Accuracy:", accuracy_loo)

# Ejecutar el código
if __name__ == "__main__":
    main()
