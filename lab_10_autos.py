import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Ruta fija del archivo
file_path = '/content/car_data.csv'

# Cargar el archivo
try:
    car_data = pd.read_csv(file_path)
    print("Archivo cargado correctamente.")
    print("Primeras filas del dataset:")
    print(car_data.head())
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# Verificar valores nulos y columnas disponibles
print("\nValores nulos por columna:")
print(car_data.isnull().sum())

print("\nColumnas disponibles:")
print(car_data.columns)

# Validar si la columna 'class' existe
if 'class' not in car_data.columns:
    print("Error: La columna 'class' no existe en el dataset.")
    exit()

# Eliminar filas con valores nulos
car_data.dropna(inplace=True)

# Codificar columnas categóricas automáticamente
categorical_columns = car_data.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    car_data[col] = le.fit_transform(car_data[col])
    label_encoders[col] = le

# Normalizar las columnas numéricas
numerical_columns = car_data.select_dtypes(exclude=['object']).columns.tolist()
scaler = StandardScaler()
car_data[numerical_columns] = scaler.fit_transform(car_data[numerical_columns])

# Separar las características (X) y la etiqueta (y)
X = car_data.drop(['class'], axis=1).values
y = car_data['class'].values

# Verificar dimensiones de X e y
print("\nDimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

# Convertir etiquetas a números si no están ya codificadas
le_class = LabelEncoder()
y = le_class.fit_transform(y)

# Función para entrenar y evaluar clasificadores
def evaluate_classifiers(X, y):
    results = {}

    # Hold-Out Validation (70/30)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    results['Hold-Out Naive Bayes'] = {
        'Accuracy': accuracy_score(y_test, y_pred_nb),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_nb).tolist()
    }
    
    # KNN (k=5)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    results['Hold-Out KNN'] = {
        'Accuracy': accuracy_score(y_test, y_pred_knn),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_knn).tolist()
    }

    # 10-Fold Cross-Validation
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    nb_cv_scores = cross_val_score(nb_model, X, y, cv=kfold)
    knn_cv_scores = cross_val_score(knn_model, X, y, cv=kfold)
    results['10-Fold Naive Bayes'] = {'Mean Accuracy': nb_cv_scores.mean()}
    results['10-Fold KNN'] = {'Mean Accuracy': knn_cv_scores.mean()}

    # Leave-One-Out Validation
   
    loo = LeaveOneOut()
    nb_loo_scores = cross_val_score(nb_model, X, y, cv=loo)
    knn_loo_scores = cross_val_score(knn_model, X, y, cv=loo)
    results['Leave-One-Out Naive Bayes'] = {'Mean Accuracy': nb_loo_scores.mean()}
    results['Leave-One-Out KNN'] = {'Mean Accuracy': knn_loo_scores.mean()}

    return results

# Evaluar clasificadores
try:
    results = evaluate_classifiers(X, y)
    # Mostrar resultados
    print("\nResultados:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
except ValueError as ve:
    print(f"Error en la evaluación: {ve}")
