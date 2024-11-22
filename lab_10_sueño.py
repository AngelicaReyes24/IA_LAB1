import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Ruta fija del archivo
file_path = '/content/student_sleep_patterns.csv'

# Leer el archivo completo
data = pd.read_csv(file_path)

# Asumimos que la última columna es la etiqueta (y), y las demás son características (X)
X = data.iloc[:, :-1].values  # Todas las columnas excepto la última
y = data.iloc[:, -1].values   # Última columna como etiqueta

# Función para entrenar y validar modelos
def evaluate_classifiers(X, y):
    results = {}
    
    # Dividir datos para Hold-Out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Clasificador Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    results['Naive Bayes'] = {
        'Accuracy': accuracy_score(y_test, y_pred_nb),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_nb).tolist()
    }
    
    # Clasificador KNN
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    results['KNN'] = {
        'Accuracy': accuracy_score(y_test, y_pred_knn),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_knn).tolist()
    }
    
    return results

# Evaluar clasificadores
results = evaluate_classifiers(X, y)

# Mostrar resultados
print("\nResultados de Hold-Out Validation:")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  Accuracy: {metrics['Accuracy']:.2f}")
    print(f"  Confusion Matrix: {metrics['Confusion Matrix']}")
