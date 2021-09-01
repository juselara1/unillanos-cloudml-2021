#%% [markdown]
# # Taller de Machine Learning en la Nube con Python - Introducción a Matplotlib
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
# ## Que es Matplotlib?
#
# Se trata de una librería para visualización. Permite crear gráficas estáticas, interactivas y animadas en Python. Es una librería que usa varios toolkits para interfaces gráficas de usuario como Qt, Tkinter, entre otros.
#
# <img src="https://matplotlib.org/_static/logo2_compressed.svg" width=500>
#
# `matplotlib` típicamente se importa utilizando el alias `plt`:

#%%
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

#%% [markdown]
# ## Objetos Base
#
# En `matplotlib` típicamente vamos a estar tratando con dos tipos de objetos: las figuras y los ejes.

#%% [markdown]
# ### Figuras
#
# Las figuras son los objetos que representan las ventanas a nivel de interfaz gráfica. Una figura puede contener varios ejes dentro de sí y se definen de la siguiente forma:

#%%
fig = plt.Figure(
    figsize=(10, 10),
    facecolor=(0.1, 0.1, 0.1)
)
print(fig)
print(type(fig))

#%% [markdown]
# ### Ejes
#
# Los ejes son objetos que representan las gráficas y generalmente se ven gráficamente como planos cartesianos o ejes 3D. Se definen de la siguiente forma:

#%% 
ax = fig.add_axes([0, 0, 1, 1]) # x_inicial, y_inicial, ancho, alto
fig

#%% [markdown]
# Una figura puede tener uno o varios ejes, en cada eje vamos a poner las visualizaciones que necesitemos generar. Veamos un ejemplo de una figura con dos ejes:

#%% 
fig = plt.Figure(
    figsize=(10, 10),
    facecolor=(0.1, 0.1, 0.1)
)
ax1 = fig.add_axes([0, 0, 0.45, 1])
ax2 = fig.add_axes([0.55, 0, 0.45, 1])

#%% [markdown]
# Cuando requerimos que los ejes tengan el mismo tamaño y se ubiquen en forma de retícula, podemos usar el método `subplots`:

#%%
fig, ax = plt.subplots(
    2, 2,
    figsize=(10, 10)
)
print(fig)
print(ax)

#%% [markdown]
# ## Tipos de Gráficas
#
# Veamos ejemplos de algunos tipos de gráficas que podemos generar en `matplotlib`.

#%% [markdown]
# ### Line Plot
#
# Se trata de un tipo de gráfico en el que se muestra una serie de puntos unidos por medio de una linea.

#%%
# definimos dos arreglos de numpy
x = np.linspace(-10, 10, 100)
y = x ** 2

fig, ax = plt.subplots(1, 1)
ax.plot(x, y)
fig.savefig("im.png")

#%% [markdown]
# ### Scatter Plot
# 
# Es una gráfica que permite visualizar una nube de puntos.

#%%
# definimos un arreglo de numpy
X = np.random.normal(size=(100, 2))
fig, ax = plt.subplots(1, 1)
ax.scatter(X[:, 0], X[:, 1])

#%% [markdown]
# ### Bar Plot
#
# Esta gráfica nos permite generar un diagrama de barras.

#%%
# definimos elementos y sus conteos:
labels = ["Perros", "Gatos", "Pajaros"]
counts = [50, 50, 10]

fig, ax = plt.subplots(1, 1)
ax.bar(labels, counts)

#%% [markdown]
# ### Pie Plot
#
# Diagrama de pastel.

#%%
labels = ["Perros", "Gatos", "Pajaros"]
freqs = [0.4, 0.4, 0.2]
fig, ax = plt.subplots(1, 1)
ax.pie(freqs, labels=labels)

#%% [markdown]
# ### Imshow
#
# Nos permite visualizar imágenes y mapas de calor.
# **Puede descargar los datos, corriendo el siguiente comando**
# ```python
# !mkdir data
# !wget https://raw.githubusercontent.com/juselara1/unillanos-cloudml-2021/master/matplotlib/data/im.npy -P data/
# ```

#%%
X = np.load("data/im.npy")
fig, ax = plt.subplots(1, 1)
ax.imshow(X, cmap="gray")

#%% [markdown]
# Ahora, veamos un ejemplo de un mapa de calor, vamos a generarlo para una distribución normal bi-variada.

#%% 
# definimos la PDF de la distribución normal
def multivariate_normal(
        x,
        mu=np.zeros((2, 1)),
        Sigma=np.eye(2)
        ):
    """
    Calculamos la PDF de una distribución normal.
    """
    ker = np.exp(
            -0.5 * 
            (x - mu).T @ 
            np.linalg.inv(Sigma) @
            (x - mu)
            )
    return float(ker / (
            2 * np.pi * 
            np.sqrt(np.linalg.det(Sigma))
            ))

# Creamos una retícula de puntos
dom = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(
        dom, dom
        )

# Concatenamos los puntos 
points = np.hstack([
    X.reshape(-1, 1),
    Y.reshape(-1, 1)
    ])

# Evaluamos la pdf en cada punto
pdf_x = np.array([
    multivariate_normal(
        point.reshape(-1, 1)
        )
    for point in points
    ])

#%% [markdown]
# Finalmente, generamos el mapa de calor

#%%
fig, ax = plt.subplots(1, 2)
ax[0].imshow(
        pdf_x.reshape(X.shape),
        cmap="Blues"
        );
ax[0].set_xticks([0, 50, 99]);
ax[0].set_xticklabels([-1, 0, 1])
ax[0].set_yticks([0, 50, 99]);
ax[0].set_yticklabels([-1, 0, 1])

ax[1].contour(
        X, Y,
        pdf_x.reshape(X.shape),
        cmap="Blues"
        );
fig.tight_layout()

