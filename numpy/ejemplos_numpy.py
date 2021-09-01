#%% [markdown]
## Taller de Machine Learning en la Nube con Python - Ejemplos de Numpy 
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
import numpy as np
import matplotlib.pyplot as plt # Librería para visualización, la veremos más adelante
# %matplotlib inline

#%% [markdown]
### Sistemas de Ecuaciones Lineales
# Supongamos que tenemos el siguiente sistema de ecuaciones lineales:
#
# $$
# a_1 x + b_1 y + c_1 z = d_1\\
# a_2 x + b_2 y + c_2 z = d_2\\
# a_3 x + b_3 y + c_3 z = d_3
# $$
#
# Donde $\{x, y, z\}$ son las variables a determinar. Esto se puede reescribir de la siguiente forma:
#
# $$\mathbf{A}\mathbf{x} = \mathbf{y}$$
#
# Es decir:
# $$
# \mathbf{A} = \left(\begin{matrix}
#   a_1 & b_1 & c_1\\
#   a_2 & b_2 & c_2\\
#   a_3 & b_3 & c_3
#   \end{matrix}\right)\\
# \mathbf{x} = \left(\begin{matrix}
#   x\\
#   y\\
#   z\\
#   \end{matrix}\right)\\
# \mathbf{y} = \left(\begin{matrix}
#   d_1\\
#   d_2\\
#   d_3\\
#   \end{matrix}\right)
# $$
#
# Si resolvemos la ecuación:
#
# $$
# \mathbf{x} = \mathbf{A}^{-1}\mathbf{y}
# $$

#%% [markdown]
# Ahora solucionemos el siguiente sistema de ecuaciones:
# $$
# 3x + 2y + z = 0\\
# -2x + 4y + 2z = 2\\
# 5x + 10y + 20z = 3 
# $$
# Comenzaremos creando los arreglos $\mathbf{A}$ e $\mathbf{y}$:

#%%
A = np.array([
    [3, 2, 1],
    [-2, 4, 2],
    [5, 10, 20]
    ])
y = np.array([
    [0],
    [2],
    [3]
    ])

#%% [markdown]
# Veamos la solución:

#%%
x = np.linalg.inv(A) @ y
print(x)

#%% [markdown]
### Regresión Lineal
# La regresión lineal es un problema en el que se busca ajustar una función que mejor describa la relación lineal entre dos conjuntos de variables: *dependientes* ($\mathbf{y}$) e *independientes* ($\mathbf{x}$).
#
# El problema en la regresión lineal es que hay cierta cantidad de ruido $\epsilon$, por lo que hay que encontrar la mejor recta capture dicha relación entre variables.
# 
# Primero, comenzamos definiendo un conjunto de parámetros $\mathbf{w}$ y unos valores estimados o predicciones $\hat{\mathbf{y}}$:
#
# $$
# \hat{\mathbf{y}} = \mathbf{X} \mathbf{w}
# $$
#
# En este caso, el problema consiste en encontrar el vector de parámetros $\mathbf{w} \in \mathbb{R}^{m \times 1}$ a partir de observaciones compuestas por $\mathbf{X} \in \mathbb{R}^{N \times m}$ variables dependientes y $\mathbf{y} \in \mathbb{R}^{N \times 1}$ variables dependientes.
#
# Comencemos cargando unos datos:
#
# **Puede descargarlos corriendo el siguiente comando**
#
# ```sh
# !mkdir data
# !wget https://raw.githubusercontent.com/juselara1/unillanos-cloudml-2021/master/numpy/data/reg.npy -P data/
# ```

#%%
data = np.load("data/reg.npy", allow_pickle=True)
X = data.item()["x"]
y = data.item()["y"]
print(X.shape)
print(y.shape)

#%% [markdown]
# Veamos como se ven estos datos:

#%%
plt.plot(X.flatten(), y.flatten())
plt.xlabel("Temperatura")
plt.ylabel("Presión")

#%% [markdown]
# Ahora veamos como determinar los parámetros $\mathbf{w}$. Comenzaremos definiendo la relación entre las variables independientes $\mathbf{y}$ y las predicciones $\hat{\mathbf{y}}$:
#
#$$
# \mathbf{y} = \hat{\mathbf{y}} + \epsilon
#$$
#
# En este caso $\epsilon$ corresponde a una variable aleatoria que captura el error entre los valores observados $\mathbf{y}$ y los predichos $\hat{\mathbf{y}}$. En este caso supondremos que el ruido tiene una distribución normal $\epsilon \sim N(0, \sigma^2)$ y buscaremos minimizar su magnitud:
#
# $$
# \mathbf{w} = \underset{\mathbf{w}}{\text{argmin}}{||\epsilon||^2}\\
# \mathbf{w} = \underset{\mathbf{w}}{\text{argmin}}{||\mathbf{y} - \hat{\mathbf{y}}||^2}
# $$
#
# Al minimizar (derivar) esta función, obtendremos:
#
# $$
# \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
# $$
#
# Veamos esta solución en numpy:

#%%
w = np.linalg.inv(X.T @ X) @ X.T @ y
print(w)

#%% [markdown]
# Veamos el resultado:

#%%
y_hat = X @ w
plt.plot(X.flatten(), y.flatten(), label="Observaciones")
plt.plot(X.flatten(), y_hat.flatten(), label="Predicciones")
plt.legend()

#%% [markdown]
# El resultado se ve bastante bien, no obstante puede mejorar si incluimos un parámetro de intercepto. Para no modificar mucho el procedimiento matemático agregaremos una columna de unos en la matriz $\mathbf{X}$.
# 
# **¿Por qué agregar una columna de unos es igual a considerar un intercepto?**

#%%
X_intercept = np.hstack([X, np.ones_like(X)])
print(X_intercept[:5])
print(X_intercept.shape)

#%% [markdown]
# Miremos el resultado con este cambio:

#%%
w_2 = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y

#%%
y_hat2 = X_intercept @ w_2
plt.plot(X.flatten(), y.flatten(), label="Observaciones")
plt.plot(X.flatten(), y_hat.flatten(), label="Predicciones")
plt.plot(X.flatten(), y_hat2.flatten(), label="Predicciones con intercepto")
plt.legend()
