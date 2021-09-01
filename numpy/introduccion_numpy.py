#%% [markdown]
## Taller de Machine Learning en la Nube con Python - Introducción a Numpy
# **Juan S. Lara**
#
# *Universidad Nacional de Colombia*
#
# [julara@unal.edu.co]()
#
# <a href="https://github.com/juselara1"><img src="https://mpng.subpng.com/20180326/gxq/kisspng-github-computer-icons-icon-design-github-5ab8a31e334e73.4114704215220498222102.jpg" width="20" align="left"></a>
# <a href="https://www.linkedin.com/in/juan-sebastian-lara-ramirez-43570a214/"><img src="https://image.flaticon.com/icons/png/512/174/174857.png" width="20" align="left"></a>
# <a href="https://www.researchgate.net/profile/Juan-Lara-Ramirez"><img src="https://user-images.githubusercontent.com/511683/28757557-f82cff1a-7585-11e7-9317-072a838dcca3.png" width="20" align="left"></a>

#%% [markdown]
### ¿Que es Numpy?
#
# Numpy es una librería de álgebra lineal para Python que permite la manipulación de vectores, matrices y arreglos multidimensionales. Está escrito principalmente en C y en Python, lo que permite una gran velocidad de cómputo con la elegante sintaxis de Python.
# 
# <img src="https://miro.medium.com/max/765/1*cyXCE-JcBelTyrK-58w6_Q.png" width="400">
# 
# Numpy se puede instalar con `pip`:
#
# ```
# pip install numpy
# ```
#
# Comencemos importando la librería, por convención le pondremos el alias `np`.

#%%
import numpy as np

#%% [markdown]
### Arreglos Multidimensionales
#
# Los arreglos multidimensionales son la estructura de datos base en Numpy, se trata de un tipo de datos que permite representar tensores. Los tensores son una forma general de estructuras matemáticas como:
#
# <img src="http://www.sr-sv.com/wp-content/uploads/2019/08/Tensor_01.jpg" width="400">
# 
# * Tensor rango 0: Escalares $\mathbb{R}^1$.
# * Tensor rango 1: Vectores $\mathbb{R}^n$.
# * Tensor rango 2: Matriz $\mathbb{R}^{n \times m}$.
# * Tensor rango d: Arreglo multidimensional $\mathbb{R}^{n_1 \times n_2 \times \dots n_d}$.
#
# Un arreglo multidimensional se define de la siguiente forma:

#%%
X = np.array([
    [10, 10],
    [15, 5]
    ]) # vector de dimensión 2
X

#%% [markdown]
# Los arreglos de numpy tienen distintos atributos, como:

#%%
print(f"tamaño {X.size}")
print(f"forma {X.shape}")
print(f"tipo {X.dtype}")

#%% [markdown]
### Métodos de los Arreglos
# Existen algunos métodos importantes que nos permiten manipular internamente cada arreglo, veamos algunos ejemplos:
# * Cambio de forma:

#%%
X_2 = X.reshape((1, 4)) # pasamos de (2, 2) a (1, 4)
print("Antes:")
print(X)
print(X.shape)
print("Después de reshape:")
print(X_2)
print(X_2.shape)

#%% [markdown]
# * Cambio de tipo:

#%%
X_f = X.astype(np.float32)
print("Antes:")
print(X)
print(X.dtype)
print("Después:")
print(X_f)
print(X_f.dtype)

#%% [markdown]
# * Transpuesto de una matriz:

#%%
X_t = X.T
print("Antes:")
print(X)
print("Después:")
print(X_t)

#%% [markdown]
# * Aplanamiento:

#%%
X_flat = X.flatten()
print("Antes:")
print(X)
print(X.shape)
print("Después:")
print(X_flat)
print(X_flat.shape)

#%% [markdown]
### Creación de Arreglos
# Existen diversas formas de crear arreglos de numpy. Veamos algunos ejemplos.
# * A partir de listas en Python:

#%%
X = np.array([1.0, 3.4])
print(X)

#%% [markdown]
# * Constantes:

#%%
# arreglo de unos
X = np.ones(
        shape=(3, 3),
        dtype=np.float64
        )
print(X)

#%%
# arreglo de ceros
X = np.zeros(
        shape=(2, 4),
        dtype=np.bool8
        )
print(X)

#%% 
# arreglo de números aleatorios
X = np.random.uniform(
        low=0, high=1,
        size=(5, 3)
        )
print(X)

#%% [markdown]
### Indexado de Arreglos
# Uno de los elementos clave en numpy es la indexación. Se trata de un método para seleccionar elementos dentro de los arreglos a través de slices o cortes.
#
# Veamos algunos ejemplos.
#
# * Selección de un elemento $X_{ij}$:

#%%
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    ])
print(X[0, 1]) # Elemento de la fila 0, columna 1

#%% [markdown]
# * Selección de la fila 2:

#%%
print(X[2, :])

#%% [markdown]
# * Selección de la columna 1:

#%%
print(X[:, 1])

#%% [markdown]
# * Selección de los elementos entre las filas 0 y 1, y las columnas 1 y 2:

#%%
print(X[:2, 1:])

#%% [markdown]
### Operaciones con Arreglos
# Numpy nos permite realizar distintos tipos de operaciones con arreglos, incluyendo sumas, promedios, productorias, entre otros. Veamos las operaciones de agregación generales:
#
# * Sumas

#%%
print(X.sum()) # suma de todos los elementos en el arreglo

#%%
print(X.sum(axis=0)) # suma por columnas

#%% 
print(X.sum(axis=1)) # suma por filas

#%% [markdown]
# * Promedios

#%%
print(X.mean()) # promedio de todos los elementos

#%%
print(X.mean(axis=0)) # promedio por columna

#%% 
print(X.mean(axis=1)) # promedio por filas

#%% [markdown]
# * Desviación estándar

#%%
print(X.std()) # desviación estándar de todos los elementos

#%%
print(X.std(axis=0)) # desviación estándar por columna

#%%
print(X.std(axis=1)) # desviación estándar por fila

#%% [markdown]
# * Operaciones algebraicas

#%%
X = np.array([
    [2, -1],
    [1, 3]
    ])
print(np.linalg.inv(X)) # inverso de una matriz.

#%%
print(np.linalg.eig(X)) # valores y vectores propios.

#%%
print(np.linalg.det(X)) # determinante de la matriz.

#%%
print(np.linalg.matrix_rank(X)) # rango de una matriz

#%% [markdown]
### Operaciones entre Arreglos
# También podemos realizar operaciones entre distintos arreglos.

#%%
X = np.random.randint(10, size=(5, 5))
Y = np.random.randint(10, size=(5, 5))

#%%
print(X + Y) # suma de matrices

#%%
print(X - Y) # resta de matrices

#%%
print(X * Y) # producto elemento a elemento

#%%
print(X @ Y) # producto matricial

#%%
print(X / Y) # división elemento a elemento
