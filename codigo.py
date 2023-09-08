"""
Autor : Josue Bernardo Villegas Nunio
Matricula: A01751694
Fecha de entrega: 06-09-2023
Nombre del trabajo: Momento de Retroalimentación: Módulo 2 Análisis y Reporte sobre el desempeño del modelo. (Portafolio Análisis)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Carga el conjunto de datos
data = pd.read_csv('VSRR_Provisional_Drug_Overdose_Death_Counts.csv', na_values=['Data not shown due to suppression or insufficient data'])

# Elimina la columna 'Period' si no es relevante
data.drop(columns=['Period'], inplace=True)

# Manejo de Comas y Caracteres no Deseados
def clean_numeric_columns(col):
    if col.dtype == 'O':  # Solo aplica a columnas de tipo objeto (texto)
        col = col.str.replace(',', '', regex=True).str.replace('%', '', regex=True).str.rstrip('_IMPORTANT')
    return col

data[['Data Value', 'Percent Complete', 'Percent Pending Investigation', 'Predicted Value']] = data[['Data Value', 'Percent Complete', 'Percent Pending Investigation', 'Predicted Value']].apply(clean_numeric_columns)

# Manejo de Valores Faltantes con SimpleImputer después de limpiar las comas
numeric_cols = ['Data Value', 'Percent Complete', 'Percent Pending Investigation', 'Predicted Value']
imputer = SimpleImputer(strategy='mean')  # Puedes ajustar la estrategia según tus necesidades
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Manejo de Comas y Caracteres no Deseados
def clean_numeric_columns(col):
    if col.dtype == 'O':  # Solo aplica a columnas de tipo objeto (texto)
        col = col.str.replace(',', '', regex=True).str.replace('%', '', regex=True).str.rstrip('_IMPORTANT')
    return col.astype(float)

data[['Data Value', 'Percent Complete', 'Percent Pending Investigation', 'Predicted Value']] = data[['Data Value', 'Percent Complete', 'Percent Pending Investigation', 'Predicted Value']].apply(clean_numeric_columns)

# Manejo de Columnas de Fecha
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'] + '-01')

# Codificacion de Columnas Categoricas
data = pd.get_dummies(data, columns=['State', 'Month', 'State Name'], drop_first=True)

# Codificacion de la columna 'Indicator' (target)
label_encoder = LabelEncoder()
data['Indicator'] = label_encoder.fit_transform(data['Indicator'])

# Division de los datos en conjuntos de entrenamiento, prueba y validacion
X = data.drop(columns=['Date', 'Indicator', 'Footnote', 'Footnote Symbol'])
y = data['Indicator']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Imprime el tamaño de cada conjunto
print(f"Tamaño del conjunto de entrenamiento: {len(X_train)} ejemplos")
print(f"Tamaño del conjunto de validación: {len(X_val)} ejemplos")
print(f"Tamaño del conjunto de prueba: {len(X_test)} ejemplos")

# Matriz de Correlación
correlation_matrix = X_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación de Características')
plt.show()

# Entrenamiento y evaluación del modelo simple (Logistic Regression)
model_simple = LogisticRegression(max_iter=1000)
model_simple.fit(X_train, y_train)
pred_simple_val = model_simple.predict(X_val)
accuracy_simple_val = accuracy_score(y_val, pred_simple_val)

# Entrenamiento y evaluación del modelo complejo (Decision Tree)
model_complex = DecisionTreeClassifier()
model_complex.fit(X_train, y_train)
pred_complex_val = model_complex.predict(X_val)
accuracy_complex_val = accuracy_score(y_val, pred_complex_val)

print(f"Precision del modelo simple en validación: {accuracy_simple_val}")
print(f"Precision del modelo complejo en validación: {accuracy_complex_val}")

# Curvas ROC y AUC
from sklearn.metrics import roc_curve, roc_auc_score

# Para el modelo simple
y_prob_simple_val = model_simple.predict_proba(X_val)[:, 1]
fpr_simple, tpr_simple, _ = roc_curve(y_val, y_prob_simple_val)
roc_auc_simple = roc_auc_score(y_val, y_prob_simple_val)

# Para el modelo complejo
y_prob_complex_val = model_complex.predict_proba(X_val)[:, 1]
fpr_complex, tpr_complex, _ = roc_curve(y_val, y_prob_complex_val)
roc_auc_complex = roc_auc_score(y_val, y_prob_complex_val)

plt.figure(figsize=(8, 6))
plt.plot(fpr_simple, tpr_simple, label='ROC Curve (Simple Model) AUC = {:.2f}'.format(roc_auc_simple))
plt.plot(fpr_complex, tpr_complex, label='ROC Curve (Complex Model) AUC = {:.2f}'.format(roc_auc_complex))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC y AUC')
plt.legend()
plt.show()

# Curva de Aprendizaje
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model_complex, X_train, y_train, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Entrenamiento')
plt.plot(train_sizes, test_mean, label='Validación')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Precisión Media')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()
