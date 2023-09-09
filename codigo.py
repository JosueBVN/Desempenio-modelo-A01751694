# Importa las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

# Deshabilita las advertencias de convergencia
import warnings
warnings.filterwarnings("ignore")

# Carga el conjunto de datos
data = load_breast_cancer()
X = data.data
y = data.target

# Divide los datos en conjuntos de entrenamiento, prueba y validación
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normaliza los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Entrena un modelo de Regresión Logística
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evalúa el modelo en los conjuntos de entrenamiento, prueba y validación
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calcula métricas de rendimiento
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calcula matriz de Confusion
confusion_train = confusion_matrix(y_train, y_train_pred)
confusion_val = confusion_matrix(y_val, y_val_pred)
confusion_test = confusion_matrix(y_test, y_test_pred)

# Muestra métricas de rendimiento y matrices de Confusion
print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

print("Matriz de Confusion en el conjunto de entrenamiento:")
print(confusion_train)

print("Matriz de Confusion en el conjunto de validación:")
print(confusion_val)

print("Matriz de Confusion en el conjunto de prueba:")
print(confusion_test)

# Plotea curvas de aprendizaje para diagnosticar bias/variance
train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 5))
plt.title("Curva de Aprendizaje")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precisión")
plt.plot(train_sizes, train_scores_mean, label="Train")
plt.plot(train_sizes, val_scores_mean, label="Validation")
plt.legend()
plt.show()

# Plotea curvas de validación para ajuste del modelo
param_range = np.logspace(-3, 3, 7)
train_scores, val_scores = validation_curve(
    model, X_train, y_train, param_name="C", param_range=param_range, cv=5
)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 5))
plt.title("Curva de Validación para Ajuste del Modelo")
plt.xlabel("Parámetro C")
plt.ylabel("Precisión")
plt.semilogx(param_range, train_scores_mean, label="Train")
plt.semilogx(param_range, val_scores_mean, label="Validation")
plt.legend()
plt.show()

# Calcula el tamaño de los conjuntos de entrenamiento y validación
total_samples = len(X)
train_size = len(X_train)
val_size = len(X_val)

# Etiquetas y tamaños de las porciones
labels = ['Entrenamiento', 'Validación']
sizes = [train_size, val_size]

# Colores de las porciones
colors = ['lightblue', 'lightgreen']

# Explode: resalta la porción de validación
explode = (0.1, 0)

# Crea la gráfica de pastel
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Asegura que el gráfico sea un círculo perfecto

# Título
plt.title('Proporción de Datos en Conjuntos de Entrenamiento y Validación')

# Muestra la gráfica
plt.show()
