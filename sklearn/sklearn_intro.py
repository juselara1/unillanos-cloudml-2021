#%% [markdown]
## Taller de Machine Learning en la Nube con Python - Introducción a Scikit Learn
# **Juan S. Lara**
#
# *Universidad Nacional de Colombia*
#
# [julara@unal.edu.co]()
#
# <a href="https://github.com/juselara1"><img src="https://mpng.subpng.com/20180326/gxq/kisspng-github-computer-icons-icon-design-github-5ab8a31e334e73.4114704215220498222102.jpg" width="20" align="left"></a>
# <a href="https://www.linkedin.com/in/juan-sebastian-lara-ramirez-43570a214/"><img src="https://image.flaticon.com/icons/png/512/174/174857.png" width="20" align="left"></a>
# <a href="https://www.researchgate.net/profile/Juan-Lara-Ramirez"><img src="https://user-images.githubusercontent.com/511683/28757557-f82cff1a-7585-11e7-9317-072a838dcca3.png" width="20" align="left"></a>

#%% 
# librerías para manipulación de datos
import numpy as np

# modelos y funcionalidades de sklearn que se van a usar
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# conjuntos de datos a usar
from sklearn.datasets import load_digits, load_boston, load_sample_image
from scipy.misc import face

# librerías de visualización
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# %matplotlib inline

#%% [markdown]
# ## ¿Qué es Scikit-learn?
# `sklearn` es una librería para la implementación, entrenamiento, evaluación y selección de modelos de machine learning. Contiene varios modelos listos para usar, métricas, técnicas de validación, creación de flujos, entre otros.
#
# <img src="https://scikit-learn.org/stable/_static/ml_map.png" width="800">
#%% [markdown] 
# ## Clasificación
#
# Un problema de clasificación consiste en partir de un conjunto de características o variables $\mathbf{X} \in \mathbb{R}^{N \times m}$ y predecir una o varias categorías $\mathbf{y} \in \mathbb{R}^{N \times k}$. Veamos un ejemplo practico.

#%% [markdown]
# ### Carga de datos
#
# Vamos a usar un conjunto de datos que contiene imágenes de números escritos a mano y su respectiva etiqueta:

#%% 
digits = load_digits()

#%% [markdown]
# Veamos un ejemplo de las imágenes:

#%% 
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
cont = 0
for i in range(3):
    for j in range(3):
        ax[i, j].imshow(digits["images"][cont], "gray")
        ax[i, j].set_title(digits["target"][cont])
        ax[i, j].axis("off")
        cont += 1

#%% [markdown] 
# ### Creación del Conjunto de Datos
# Comenzaremos definiendo el conjunto de datos como arreglos de numpy:

#%%
X = digits["data"]
y = digits["target"]
print(X.shape)
print(y.shape)

#%% [markdown]
# ### Ahora tomaremos un enfoque típico para validación de modelos conocido como cross-validation. Consiste en dividir el conjunto de datos en dos partes:
# * Conjunto de entrenamiento: datos sobre los que se va a entrenar el modelo o a realizar la estimación de parámetros.
# * Conjunto de validación: datos sobre los que se evalúa el modelo una vez entrenado.
# Veamos como hacer esto con `sklearn`

#%%
X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3
        )
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#%% [markdown]
# ### Modelamiento
#
# Existen diversos modelos que pueden ser usados para clasificación, en este caso nos enfocaremos mas en la metodología que en el detalle matemático. Para la demostración, utilizaremos una red neuronal (veremos sus detalles mas adelante).

#%% 
model = MLPClassifier((32, 32))

#%% [markdown]
# Ahora, veamos como es el entrenamiento o estimación de parámetros: 

#%% 
model.fit(X_train, y_train)

#%% [markdown]
# Veamos algunos ejemplos de lo que aprendió el modelo

#%% 
idx = 10 # podemos cambiar este valor para ver diferentes imágenes
plt.imshow(
        X_train[idx].reshape(8, 8),
        cmap="gray"
        )
plt.title(
        f"Predicción: {model.predict(X_train[idx:idx + 1, :])[0]}"
        );
plt.axis("off");

#%% 
# ### Evaluación del modelo
#
# La evaluación del modelo la realizaremos sobre el conjunto de evaluación (datos no vistos) de esta forma determinaremos que tanto esta generalizando el modelo entrenado. Primero evaluemos la exactitud:

#%%
model.score(X_test, y_test) # Evaluamos la exactitud del modelo (numero de aciertos)

#%% [markdown]
# Esto lo podemos ver de forma más detallada con una matriz de confusión:

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ax.imshow(cm);
ax.set_xlabel("Predicho");
ax.set_ylabel("Real");
ax.set_xticks(range(10));
ax.set_yticks(range(10));

#%% [markdown]
# ## Regresión
#
# Ahora vamos a ver un problema similar en el que no tenemos que predecir una categoría sino un valor continuo (decimal).

#%% [markdown] 
# ### Carga de datos
#
# Vamos a crean un conjunto de datos

#%% 
x = np.linspace(0, 10, 1000)
y = np.sin(2 * x) * np.exp(-0.1 * x) + np.random.normal(scale=0.1, size=x.shape)
print(x.shape)
print(y.shape)
#%% 
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")

#%% [markdown]
# ### Creación del Conjunto de datos
#
# De la misma forma que en el caso anterior, vamos a dividir el conjunto de datos en dos particiones de entrenamiento y de prueba.

#%%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#%% [markdown]
# ### Modelo de Machine Learning
#
# Primero, vamos a entrenar una red neuronal con sólo una variable de entrada:

#%% 
model = MLPRegressor((128, 128)) 

#%% [markdown] 
# Entrenamos el modelo

#%%
model.fit(X_train, y_train)

#%% [markdown]
# ### Evaluación del modelo
#
# Veamos el comportamiento del modelo:

#%%
plt.scatter(
    X_test.flatten(),
    y_test,
    c="b",
    label="Original",
    alpha=0.3
)
plt.plot(
    np.sort(X_test.flatten()),
    model.predict(
        np.sort(X_test, axis=0)
    ),
    "r", label="Predicho"
)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

#%% [markdown] 
# Podemos usar como métrica el coeficiente de indeterminación $r^2$ para determinar que tan bueno fue el desempeño.

#%%
r2_score(y_test, model.predict(X_test))

#%% [markdown]
# ## Agrupamiento
#
# Ahora, veamos un ejemplo de aprendizaje no supervisado, es decir, no tenemos etiquetas $\mathbf{y}$.

#%% [markdown]
# ### Carga de datos
#
# Vamos a cargar una imagen en blanco y negro:

#%%
im = face(gray=True).astype("float32") / 255
plt.imshow(im, "gray")
plt.axis("off")

#%% [markdown]
# ### Creación del Conjunto de Datos
#
# Vamos a crear un conjunto de datos a partir de esta imagen:

#%%
X = im.reshape(-1, 1)
print(X.shape)

#%% [markdown]
# ### Modelo de Machine Learning
#
# Ahora, definimos un modelo de agrupamiento

#%% 
k = 3
model = KMeans(n_clusters=k)

#%% [markdown]
# Entrenamos el modelo

#%% 
model.fit(X)

#%% [markdown]
# ### Evaluación del modelo
#
# Veamos como son los grupos que encuentra el modelo:

#%%
preds = model.predict(X)
pred_im = preds.reshape(im.shape)

#%%
plt.figure(figsize=(10, 7))
plt.subplot(121)
plt.imshow(im, "gray")
plt.axis("off")
plt.title("Original")

plt.subplot(122)
plt.imshow(pred_im, cmap="rainbow", interpolation='nearest')
plt.axis("off")
plt.title("Grupos")
