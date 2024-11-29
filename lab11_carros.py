import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Cargar el dataset desde la ruta específica
car_data_path = '/content/car_data.csv'
car_data = pd.read_csv(car_data_path)

# Manejar valores faltantes eliminando filas con NaN
car_data_cleaned = car_data.dropna()

# Seleccionar características (X) y etiquetas (y)
X = car_data_cleaned[['city_mpg', 'highway_mpg', 'cylinders', 'displacement']].values
y = car_data_cleaned['class'].values

# Codificar etiquetas categóricas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Definir las funciones para evaluación y validación
def evaluate_model(clf, X, y, validation_type='Hold Out'):
    if validation_type == 'Hold Out':
        print("\n[Validacion: Hold Out 70/30]")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_metrics(y_test, y_pred)
        
    elif validation_type == '10-Fold Cross Validation':
        print("\n[Validation: 10-Fold Cross Validation]")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
        print(f"Precisión de Validación Cross: {accuracies}")
        print(f"Precisión promedio: {np.mean(accuracies):.4f}")

    elif validation_type == 'Leave One Out':
        print("\n[Validacion: Leave One Out]")
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_true.append(y_test[0])
            y_pred.append(clf.predict(X_test)[0])
        print_metrics(np.array(y_true), np.array(y_pred))

def print_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Matriz de Confusión:\n{cm}")
    print(f"Precisión: {accuracy:.4f}")

# Función principal para probar el dataset
def test_with_dataset(X, y):
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clasificadores
    mlp_clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, tol=1e-4, random_state=42)
    rbf_clf = SVC(kernel='rbf', random_state=42)

    print("\n Perceptron Multicapa")
    evaluate_model(mlp_clf, X_scaled, y, validation_type='Hold Out')
    evaluate_model(mlp_clf, X_scaled, y, validation_type='10-Fold Cross Validation')
    evaluate_model(mlp_clf, X_scaled, y, validation_type='Leave One Out')

    print("\n Red Neuronal RBF ")
    evaluate_model(rbf_clf, X_scaled, y, validation_type='Hold Out')
    evaluate_model(rbf_clf, X_scaled, y, validation_type='10-Fold Cross Validation')
    evaluate_model(rbf_clf, X_scaled, y, validation_type='Leave One Out')

# Probar el dataset cargado
test_with_dataset(X, y_encoded)
