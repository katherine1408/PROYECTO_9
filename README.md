# 📶 Sprint 9 – Modelo de Clasificación de Planes Móviles (Megaline)

## 📌 Descripción del Proyecto

Este proyecto pertenece al Sprint 9 del bootcamp de Ciencia de Datos y representa una introducción práctica al **Machine Learning supervisado**.

Trabajamos con datos históricos de comportamiento mensual de clientes de **Megaline**, una compañía de telefonía móvil. El objetivo es crear un modelo que recomiende el mejor plan para cada cliente: **Ultra** o **Smart**.

## 🎯 Propósito

- Desarrollar un modelo de clasificación que prediga el plan óptimo para un cliente.
- Evaluar distintos algoritmos y comparar su desempeño.
- Afinar hiperparámetros y aplicar validación cruzada.
- Cumplir un umbral mínimo de exactitud del **75%**.

## 📁 Dataset utilizado

- `users_behavior.csv`

Columnas del dataset:

- `calls`: número de llamadas realizadas en el mes.
- `minutes`: duración total de las llamadas.
- `messages`: número de SMS enviados.
- `mb_used`: tráfico de internet consumido (en MB).
- `is_ultra`: clase objetivo (1 = Ultra, 0 = Smart).

## 🧰 Funcionalidad del Proyecto

### 🔍 Procesamiento y partición
- Carga y análisis exploratorio inicial del dataset.
- División en conjunto de entrenamiento, validación y prueba.

### 🤖 Modelado
- Pruebas con diferentes modelos: DecisionTree, RandomForest, LogisticRegression.
- Ajuste de hiperparámetros.
- Comparación de métricas: precisión (`accuracy`), matriz de confusión y prueba de cordura.

### 📈 Evaluación
- Validación cruzada.
- Evaluación final sobre el conjunto de prueba.
- Selección del modelo con mejor desempeño general.

## 📊 Herramientas utilizadas

- Python  
- pandas  
- scikit-learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`)  
- matplotlib / seaborn  

---

📌 Proyecto desarrollado como parte del Sprint 9 del programa de Ciencia de Datos en **TripleTen**.
