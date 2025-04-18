
# # Megaline

# ## Análisis exploratorio de datos (Python):

# ### Inicialización:

# In[1]:


# Librerías necesarias

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ### Cargar Datos:

# In[2]:


# Cargar el dataset
data_cruda = pd.read_csv('users_behavior.csv')




# ### Visualización de datos:

# In[3]:


# Inspección inicial
display(data_cruda.head())




# ### Estudiar los datos que contienen:

# In[4]:


data_cruda.info()


# In[5]:


data_cruda.describe()


# ## Preparar los datos:

# ### Revisión de datos nulos:

# In[6]:


data_cruda.isna().sum()


# ### Revisión de datos duplicados:

# In[7]:


data_cruda.duplicated().sum()


# ## Analisis de Datos:

# ### Segmentación de datos:

# #### Entrenamiento:

# **- La función devuelve dos conjuntos:** 
# 
#      - train_full: este es el conjunto de entrenamiento con el 80% de los datos.
#      - test: este es el conjunto de prueba con el 20% de los datos.  
#      
# **- Ambos serán aleatorios, lo que significa que cada vez que ejecutes la división, podrías obtener un conjunto diferente**

# In[8]:


train_full, test = train_test_split(data_cruda, test_size=0.2)


# In[9]:


train_full


# In[10]:


test


# #### Validación:

# **- La función devuelve dos conjuntos:**  
# 
#    - train: este es el conjunto de entrenamiento, que se usará para entrenar el modelo.  
#      
#    - validacion: este es el conjunto de validación, que se usará para evaluar el modelo durante el ajuste de hiperparámetros o para verificar su rendimiento mientras se entrena.  
#      
#    - test_size=0.25 significa que el 25% de `train_full` se utilizará para el conjunto de validación, y el 75% restante se utilizará para el conjunto de entrenamiento.

# In[11]:


train, validacion = train_test_split(train_full, test_size=0.25)


# In[12]:


validacion


# #### Prueba:

# In[13]:


caracteristicas_entrenamiento = train.drop('is_ultra', axis = 1)


#   - Esta es la nueva variable que contendrá el DataFrame resultante después de haber eliminado la columna 'is_ultra'.
#   - Contendrá solo las características que se usarán para entrenar el modelo.

# In[14]:


objetivo_entrenamiento = train['is_ultra']


# Esta línea de código está extrayendo la variable objetivo del conjunto de datos de entrenamiento para que puedas usarla más adelante en el proceso de entrenamiento del modelo. En otras palabras, estás preparando los datos que el modelo intentará predecir.

# In[15]:


caracteristicas_validación = validacion.drop('is_ultra', axis = 1)


# Esta línea de código está preparando el conjunto de datos de validación al eliminar la columna que contiene las etiquetas (`is_ultra`). El resultado, `caracteristicas_validación`, contendrá solo las características que se utilizarán para hacer predicciones en el conjunto de validación.

# In[16]:


objetivo_validación = validacion['is_ultra']


# La línea de código está extrayendo la columna `is_ultra` del DataFrame `validacion` y almacenándola en la variable `objetivo_validación`. Esto significa que `objetivo_validación` ahora contiene los valores reales que se utilizarán para evaluar el rendimiento del modelo en el conjunto de validación.

# In[17]:


caracteristicas_prueba = test.drop('is_ultra', axis = 1)


# La línea de código está eliminando la columna `is_ultra` del DataFrame `test` y almacenando el resultado en `caracteristicas_prueba`. Esto significa que `caracteristicas_prueba` ahora contiene solo las columnas que se utilizarán como entradas para el modelo.

# In[18]:


objetivo_prueba = test['is_ultra']


# La línea de código está extrayendo la columna `is_ultra` del DataFrame `test` y almacenando el resultado en `objetivo_prueba`. Esto significa que `objetivo_prueba` ahora contiene la variable objetivo que se utilizará para comparar las predicciones realizadas por el modelo.

# In[19]:


print(caracteristicas_entrenamiento.shape)


# In[20]:


print(caracteristicas_validación.shape)


# In[21]:


print(caracteristicas_prueba.shape)




# #### Arbol de decision:

# El código está diseñado para evaluar cómo la profundidad de un árbol de decisión afecta su rendimiento en conjuntos de datos de prueba y validación. Esto permite identificar la profundidad óptima que maximiza la precisión sin causar sobreajuste.

# In[22]:


for profundidad in range(1,15):
    
    modelo_arbol_decision = DecisionTreeClassifier(max_depth=profundidad, random_state=12345)
    modelo_arbol_decision.fit(caracteristicas_entrenamiento,objetivo_entrenamiento)
    print('Profundidad:',profundidad)
    print('Train:',modelo_arbol_decision.score(caracteristicas_prueba,objetivo_prueba))
    print('Validacion:',modelo_arbol_decision.score(caracteristicas_validación,objetivo_validación))
    print()




# #### Modelo de regresión Logística:

# Este código crea y entrena un modelo de regresión logística, hace predicciones sobre los datos de entrenamiento y validación, y luego calcula e imprime varias métricas de rendimiento para evaluar la efectividad del modelo. 

# In[23]:


# Crear el modelo de regresión logística
modelo_regresion_logistica = LogisticRegression(random_state=12345, max_iter=1000)

# Entrenar el modelo
modelo_regresion_logistica.fit(caracteristicas_entrenamiento, objetivo_entrenamiento)

# Predicciones
predicciones_entrenamiento_lr = modelo_regresion_logistica.predict(caracteristicas_entrenamiento)
predicciones_validacion_lr = modelo_regresion_logistica.predict(caracteristicas_validación)


# - Aquí se calculan y se imprimen varias métricas de evaluación:
#   - **Exactitud**: Proporción de predicciones correctas sobre el total de predicciones, tanto para el conjunto de entrenamiento como para el de validación.
#   - **Precisión**: Proporción de verdaderos positivos sobre el total de positivos predichos. Indica cuántas de las predicciones positivas fueron realmente correctas.
#   - **Recall (Sensibilidad)**: Proporción de verdaderos positivos sobre el total de positivos reales. Mide cuántos de los positivos fueron correctamente identificados.
#   - **F1 Score**: Media armónica entre la precisión y el recall. Es útil cuando hay un desbalance en las clases y se quiere un solo número que represente el rendimiento del modelo.

# In[24]:


# Métricas para regresión logística
print("Resultados para Regresión Logística:")
print(f'Exactitud Entrenamiento: {accuracy_score(objetivo_entrenamiento, predicciones_entrenamiento_lr):.4f}')
print(f'Exactitud Validación: {accuracy_score(objetivo_validación, predicciones_validacion_lr):.4f}')
print(f'Precisión: {precision_score(objetivo_validación, predicciones_validacion_lr):.4f}')
print(f'Recall: {recall_score(objetivo_validación, predicciones_validacion_lr):.4f}')
print(f'F1 Score: {f1_score(objetivo_validación, predicciones_validacion_lr):.4f}')
print()


# #### Comparación de Metricas:

# En resumen, este código crea y entrena un modelo de árbol de decisión, hace predicciones sobre los datos de entrenamiento y validación, y luego calcula e imprime varias métricas de rendimiento para evaluar la efectividad del modelo. Esto permite comparar el rendimiento del árbol de decisión con el de otros modelos, como la regresión logística, que se había analizado previamente.

# In[25]:


# Comparar con el árbol de decisión
modelo_arbol_decision = DecisionTreeClassifier(max_depth=5, random_state=12345)
modelo_arbol_decision.fit(caracteristicas_entrenamiento, objetivo_entrenamiento)

# Predicciones del árbol de decisión
predicciones_entrenamiento_dt = modelo_arbol_decision.predict(caracteristicas_entrenamiento)
predicciones_validacion_dt = modelo_arbol_decision.predict(caracteristicas_validación)

# Métricas para árbol de decisión
print("Resultados para Árbol de Decisión:")
print(f'Exactitud Entrenamiento: {accuracy_score(objetivo_entrenamiento, predicciones_entrenamiento_dt):.4f}')
print(f'Exactitud Validación: {accuracy_score(objetivo_validación, predicciones_validacion_dt):.4f}')
print(f'Precisión: {precision_score(objetivo_validación, predicciones_validacion_dt):.4f}')
print(f'Recall: {recall_score(objetivo_validación, predicciones_validacion_dt):.4f}')
print(f'F1 Score: {f1_score(objetivo_validación, predicciones_validacion_dt):.4f}')
print()


# # Conclusiones:

# En el proyecto de clasificación para la empresa Megaline, se buscó desarrollar un modelo que pudiera predecir correctamente el plan adecuado (Smart o Ultra) para sus usuarios, utilizando datos de comportamiento como llamadas, mensajes, minutos de uso y tráfico de Internet. El objetivo era alcanzar una exactitud mínima de 0.75 en la predicción de los planes.
# 
# Se probaron dos modelos principales: **Regresión Logística** y **Árbol de Decisión**.
# 
# - La **Regresión Logística** presentó una exactitud en los conjuntos de entrenamiento y validación de 0.7070 y 0.7030, respectivamente. Aunque la precisión fue aceptable (0.7778), el modelo mostró un rendimiento deficiente en cuanto a la identificación de usuarios del plan Ultra, con un recall de 0.0357 y un F1 score de 0.0683. Estos resultados indican que este modelo tiene dificultades para detectar correctamente a los usuarios del plan Ultra, lo que lo hace menos adecuado para la tarea.
# 
# - El **Árbol de Decisión**, por otro lado, superó las expectativas al lograr una exactitud en el conjunto de validación de 0.7900, superior al umbral mínimo requerido. Además, presentó un buen equilibrio entre precisión (0.8144), recall (0.4031) y F1 score (0.5392). Esto demuestra que este modelo no solo predice de manera confiable los usuarios que pertenecen al plan Ultra, sino que también ofrece una mejora notable en la identificación de estos usuarios en comparación con la Regresión Logística.
# 
# **En resumen**, el **Árbol de Decisión** es el modelo más adecuado para la tarea de clasificación planteada por Megaline, ya que cumple con el requisito de exactitud y presenta un mejor balance entre precisión y recall, logrando así una recomendación más precisa de los planes para los usuarios. La Regresión Logística, aunque precisa en algunos aspectos, no es lo suficientemente robusta para este caso en particular.


