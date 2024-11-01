import pandas as pd

# Cambia la ruta al archivo si es necesario
file_path = "ruta/al/archivo/bezdekIris.data"

# Leer el archivo en un dataframe
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(file_path, header=None, names=column_names)

# Mostrar las primeras filas del dataframe
print(df.head())
