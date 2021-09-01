#%% [markdown]
## Taller de Machine Learning en la Nube con Python - Introducción a Tensorflow
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
# Definiremos algunas funciones para facilitar la visualización, usaremos estas funciones a lo largo del notebook para comprender los conceptos detrás de tensorflow.

#%% 
# Función para mostrar regiones de decisión
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_region(X, y, model):
    plt.figure(figsize=(7, 7))
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    min_x = min_x - (max_x - min_x) * 0.05
    max_x = max_x + (max_x - min_x) * 0.05
    min_y = min_y - (max_y - min_y) * 0.05
    max_y = max_y + (max_y - min_y) * 0.05
    x_1 = np.linspace(min_x, max_x, 100)
    x_2 = np.linspace(min_y, max_y, 100)
    x1, x2 = np.meshgrid(x_1, x_2)
    X_grid = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)
    y_pred = model.predict(X_grid)
    Z = y_pred.reshape(x1.shape)
    plt.contourf(x1, x2, Z, cmap=plt.cm.RdBu, alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y,alpha=0.7, cmap=plt.cm.RdBu, s=100)
    plt.xlabel("$x_1$");plt.ylabel("$x_2$")


#%% 
import tensorflow as tf

#%% [markdown] 
# ## ¿Qué es Tensorflow?
# Tensorflow es una plataforma end-to-end de uso libre para machine learning (ML). Se compone de un ecosistema exhaustivo y flexible de herramientas, librerías y recursos de la comunidad que permite a investigadores y desarrolladores construir e implementar aplicaciones basadas en ML.
#
# Unas de las principales ventajas de tensorflow son:
#
# |Fácil construcción de modelos|Producción robusta de ML|Experimentación para investigación|
# |---|---|---|
# |<img src="https://www.tensorflow.org/site-assets/images/marketing/home/model.svg" width="80%"> | <img src="https://www.tensorflow.org/site-assets/images/marketing/home/robust.svg" width="80%" />|<img src="https://www.tensorflow.org/site-assets/images/marketing/home/research.svg" width="80%" />|
# | Facilita la implementación y el entrenamiento <br/> de modelos de ML utilizando APIs de alto nivel <br/> como *Keras* con *eager execution*, lo cual, <br/> permite una fácil construcción y depuración.| Permite entrenar e implementar fácilmente <br/> modelos en la nube, en dispositivos móviles o <br/>en un navegador sin importar el lenguaje usado.| Tiene una arquitectura simple y flexible para <br/> llevar nuevas ideas desde el concepto hasta el <br/>código, permite usar los modelos más recientes <br/>en el estado del arte para publicar más rápido.
# |

#%% [markdown] 
# ## Acelerando el entrenamiento de los Modelos de deep learning
# El aprendizaje profundo o deep learning se refiere a una familia de modelos basados en redes neuronales artificiales que son escalables (funcionan bien con grandes cantidades de datos) y tienen gran efectividad en diferente tareas complejas sobre datos de diferente naturaleza.  
#
# Muchos de los frameworks de deep learning actuales muestran como una de sus principales características su habilidad para escalar. Esta habilidad es esencial dado que los modelos de deep learning presentan estas dos características:
#
# * Son modelos con gran cantidad de parámetros a aprender.
# * Se aplican sobre conjuntos de datos enormes.
#
# Estas características de los modelos de deep learning hacen que un entrenamiento convencional usando solamente una CPU, con varios núcleos, sea extremadamente lento. Por ello se han desarrollado varias estrategias para acelerar dicho entrenamiento entre estas se encuentra el uso de GPU (Graphic processing Unit) o unidades de procesamiento gráfico donde miles de procesadores pueden ejecutar tareas en paralelo acelerando así el entrenamiento.
#
# **De la computación gráfica al procesamiento numérico general GPGPU**
# * Instrucción única, arquitectura de datos múltiples
# * Alto rendimiento en báculos usando paralelismo de datos.
# * Hardware básico
# * Dos vendedores principales: Nvidia, AMD

#%% [markdown] 
# ## 2. Conceptos Generales en Tensorflow
#
# Veamos algunos conceptos básicos de tensorflow:

#%% [markdown] 
# ### Grafo computacional
#
# Tensorflow es de gran utilidad en *Deep learning* debido fundamentalmente a que permite realizar diferenciación automática y paraleliza operaciones matemáticas, Tensorflow consigue esto al construir internamente un grafo computacional:
#
# <img src="https://github.com/tensorflow/docs/blob/master/site/en/guide/images/intro_to_graphs/two-layer-network.png?raw=1" width="70%" />
#
# Este grafo define un flujo de datos basado en expresiones matemáticas. Más específicamente, Tensorflow utiliza un grafo dirigido donde cada nodo representa una operación o variable.
#
# Una de las principales ventajas de usar un grafo computacional, es que, las operaciones se definen como relaciones o dependencias, lo cual, permite que los cómputos sean fácilmente simplificados y paralelizados. Esto es mucho más práctico en comparación con un programa convencional donde las operaciones se ejecutan de forma secuencial.

#%% [markdown] 
# ### Tensores
#
# La principal estructura de datos utilizada en Tensorflow son los tensores. Se trata de arreglos multidimensionales que permiten guardar información. Se pueden ver como una generalización de los escalares (0D-tensor), vectores (1D-tensor) y las matrices (2D-tensor). Veamos algunos ejemplos de tensores de distintos órdenes:

#%% 
# definimos un 1D-tensor (vector) constante a partir de una lista
t = tf.constant([2, 3, 4, 5], dtype=tf.int32)
print(t)

#%% [markdown] 
# Un tensor tiene dos propiedades básicas: su forma (`shape`) y su tipo (`dtype`). Por un lado, el `shape`, al igual que en `numpy`, indica el orden, número de dimensiones, y el tamaño de cada dimensión. En el ejemplo anterior tenemos un tensor de orden 1, es decir una única dimensión, de tamaño 4. 
# Por otro lado, al igual que en cualquier lenguaje de programación, los tensores tienen un typo de representación interna: `tf.int32`, `tf.float32`, `tf.string`, entre otros. Una correcta selección del tipo de datos puede hacer los códigos más eficientes. En el ejemplo anterior, el tipo del tensor es entero de 32 bits.
#
# El siguiente ejemplo corresponde a un tensor de orden 2, una matriz, cuyo tipo es flotante de 32 bits.

#%% 
# definimos un 2D-tensor (matriz) variable a partir de una lista
t = tf.constant([[9, 5], [1, 0]], dtype=tf.float32)
print(t)

#%% [markdown] 
# En Tensorflow hay dos tipos principales de tensores:
#
# * ```tf.constant```: son arreglos multidimensionales inmutables, es decir, son tensores que no van a cambiar durante la ejecución.
# * ```tf.Variable```: se trata de tensores cuyos valores pueden cambiar durante la ejecución (por ejemplo, los parámetros de un modelo se definen como variables, ya que, estos valores se actualizan de forma iterativa).
#
# Veamos un ejemplo de variables en tensorflow:

#%% 
# definimos un 2D-tensor (matriz) variable a partir de una lista
t = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
print(t)

#%% 
# al tensor variable le podemos asignar un nuevo valor
t.assign([[-2, -1], [-3, -7]])
print(t)

#%% 
# también podemos sumarle o restarle un valor
t.assign_add([[1, 1], [1, 1]])
print(t)
t.assign_sub([[2, 2], [2, 2]])
print(t)

#%% [markdown] 
# Podemos realizar diversas operaciones y definir funciones sobre tensores, así mismo, tensorflow provee un *slicing* similar al de los arreglos de numpy. Veamos un ejemplo:

#%% 
# Definimos un 2D-tensor A
A=tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
# Definimos un 2D-tensor B
B=tf.constant([[-1, -2], [-3, -4]], dtype=tf.float32)

#%% 
# suma
A + B

#%% 
# resta
A - B

#%% 
# multiplicación por un escalar (definido en Python)
3 * A

#%% id="VtyAyKv7m-0a"
# multiplicación elemento a elemento
A * B

#%% 
# multiplicación matricial
print(tf.matmul(A, B))

#%% 
# Ejemplos de slicing
print('Tensor original:\n {}'.format(A))
# seleccionamos la primera fila
print('Primera fila:\n {}'.format(A[0]))
# seleccionamos el primer elemento de la primera fila
print('Primer elemento de la primera fila: \n {}'.format(A[0, 0]))
# Selecionamos la segunda columna
print('Segunda columna:\n {}'.format(A[:, 1]))
# Invertimos las filas
print('Filas invertidas:\n {}'.format(A[::-1]))

#%% [markdown]
# También podemos aplicar diferentes funciones matemáticas a todos los elementos de un tensor:

#%% 
# logaritmo 
tf.math.log(A)

#%% [markdown] 
# Otros tipos de operaciones aritméticas, funciones matemáticas y operaciones de álgebra lineal se pueden encontrar en el paquete ```tf.math``` y de álgebra lineal en el paquete ```tf.linalg```

#%% [markdown] 
# ### Eager execution
#
# *Tensorflow* provee un ambiente de programación imperativo (*Eager execution*) para evaluar las operaciones de forma inmediata sin la necesidad de que el usuario especifique explícitamente un grafo. Es decir, el resultado de las operaciones son valores concretos en lugar de variables simbólicas dentro del grafo computacional. Además, también permite construir el grafo de forma automática en casos donde sea requerido. Esto permite comenzar más fácilmente a programar en *Tensorflow* y a depurar modelos. Adicionalmente, *Eager execution* soporta la mayoría de las funcionalidades de *Tensorflow* y también permite aceleración por GPU.
#
# *Eager execution* es una plataforma flexible para la investigación y la experimentación que provee:
#
# * **Interfaz intuitiva**: permite desarrollar código de forma natural y usar estructuras de datos de Python. También permite desarrollar rápido aplicaciones en casos con modelos pequeños y pocos datos.
#
# * **Depuración simple**: ejecutar las operaciones directamente permite revisar a detalle los modelos durante ejecución y evaluar cambios. Además, utiliza herramientas de depuración nativas en Python para reportar errores de forma inmediata.
#
# * **Control natural**: controlar las variables desde Python en lugar de un control por medio de un grafo simplifica la especificación de modelos más dinámicos.
#
# La versión 2.0 de *Tensorflow* trae por defecto *Eager execution*.

#%% 
# Revisamos la versión de tensorflow
tf.__version__

#%% 
# Revisamos si eager execution está activa
tf.executing_eagerly()

#%% [markdown] 
# Por defecto, *Eager execution* ejecuta las operaciones de forma secuencial, es decir, no construye un grafo computacional de no ser que sea necesario para alguna operación o de ser especificado. Para que tensorflow construya el grafo debemos utilizar el decorador ```tf.function``` como se muestra a continuación:

#%%
# Definimos una función decorada (construye internamente el grafo computacional)
@tf.function
def poly(x):
    y1 = 2 * x + 3 * x ** 4 + 5 * x ** 2
    y2 = 5 * x + - 2 * x ** 4 + - 3 * x ** 2
    y3 = x + 2
    return  y1 + y2 + y3

#%% 
# Definimos una función normal en Python (ejecuta las operaciones de forma secuencial)
def poly2(x):
    y1 = 2 * x + 3 * x ** 4 + 5 * x ** 2
    y2 = 5 * x + - 2 * x ** 4 + - 3 * x ** 2
    y3 = x + 2
    return  y1 + y2 + y3

#%% 
poly2

#%% [markdown] 
# Ahora veamos una comparación entre el tiempo promedio de estas dos funciones:

#%% 
# %%timeit -n 1000
poly(tf.constant([1, 2, 3, 4], dtype=tf.float32))

#%% 
# %%timeit -n 1000
poly2(tf.constant([1, 2, 3, 4], dtype=tf.float32))

#%% [markdown]
# ### Integración con numpy
#
# Una de las principales ventajas de Tensorflow 2.0 es su compatibilidad con arreglos y operaciones de numpy. Esta última es la librería de álgebra lineal más usada en python.
#
# Veamos algunos ejemplos con numpy y Tensorflow:

#%% 
# Definimos un arreglo en numpy, la función linspace crea una secuencia de 'num' números
# igualmente distanciados entre dos límites 'start' y 'stop' 
x = np.linspace(start=0, stop=1, num=10)
x

#%% 
# Realizamos algunas operaciones en Tensorflow
acum = tf.reduce_sum(x)
acum

#%% 
# Definimos un tensor en Tensorflow
x = tf.linspace(0.0,1.0,10)
x

#%% 
# Realizamos algunas operaciones en numpy
acum = np.sum(x)
acum

#%% 
# Convertir arreglo de numpy a tensor
x = np.linspace(0,1,10)
t = tf.constant(x)
t

#%% 
# Convertir tensor a arreglo de numpy
t = tf.linspace(0.0,1.0,10)
x = t.numpy()
x

#%% [markdown] 
# En general, la sintáxis de Tensorflow es muy similar a la de numpy y dada está funcionalidad, podemos usar arreglos de cualquiera de los dos frameworks. No obstante, es necesario tener en cuenta que numpy está diseñado para trabajar con la CPU, mientras que Tensorflow permite utilizar hardware para acelerar el cómputo (e.g., unidades de procesamiento gráfico (GPU) o unidades de procesamiento tensorial (TPU)). Por lo cual, no es recomendable mezclar estos dos frameworks (la conversión de información entre la vRAM de la GPU y la RAM del computador es una operación costosa) a no ser que sea completamente necesario.
#
# Veamos un ejemplo de cómo seleccionar el tipo de hardware que se usará y una comparación del desempeño. 
#
# En el siguiente código especificamos que los cálculos deben hacerse utilizando la CPU:

#%% 
# %%timeit
# Ejecución en CPU
with tf.device("CPU:0"):
    X = tf.random.uniform(shape=(1000, 1000), minval=0, maxval=1)
    # La siguiente función evalúa si un tensor está en determinado dispositivo
    assert X.device.endswith("CPU:0")
    tf.matmul(X, X)

#%% [markdown] 
# El siguiente código especifica que el dispositivo para llevar a cabo la ejecución debe ser la GPU:

#%% 
# %%timeit
# Ejecución en GPU
with tf.device("GPU:0"):
    X = tf.random.uniform(shape=(1000, 1000), minval=0, maxval=1)
    # La siguiente función evalúa si un tensor está en determinado dispositivo
    assert X.device.endswith("GPU:0")
    tf.matmul(X, X)

#%% [markdown] 
# Ahora, veamos una comparación de la misma operación en numpy:

#%% 
# %%timeit
# Ejecución en CPU con numpy
X = np.random.uniform(0, 1, size=(1000, 1000))
X @ X

#%% [markdown] 
# ### Keras
#
# Originalmente, Keras era un *framework* de alto nivel escrito en Python que utilizaba diferentes *backends* de *deep learning* como: Tensorflow, CNTK o Theano. Actualmente, es un paquete dentro de Tensorflow 2.0 que nos permite simplificar tanto el entrenamiento como el diseño de modelos de *machine learning* y redes neuronales. 
#
# ```tf.keras``` es usado para la creación rápida de modelos y tiene tres ventajas:
#
# * **Amigable al usuario**: keras tiene una interfaz simple y consistente que ha sido optimizada para el uso en casos típicos. 
# * **Modular**: la construcción de modelos se basa en conectar bloques personalizables con pocas restricciones.
# * **Fácil extensión**: permite implementar nuevos módulos fácilmente usando todas las funcionalidades de Tensorflow, lo cual, facilita la construcción de nuevos modelos o modelos del estado del arte.
#
# Veamos ejemplos con modelos básicos de *machine learning* usando keras.

#%% [markdown] 
# #### Regresión lineal
#
# Comencemos con la implementación de uno de los modelos más simples de *machine learning*: el modelo lineal o la regresión lineal. Para ello, utilizaremos un dataset que contiene información de la edad y la estatura en niños. 

#%% 
# Cargar el dataset
X = tf.random.uniform(
        shape=(100, 1),
        minval=0,
        maxval=20,
        dtype=tf.float32
        )
y = X * 8 + 50 + tf.random.normal(
        shape=(100, 1),
        mean=0,
        stddev=10,
        dtype=tf.float32
        )

# Visualicemos los datos
plt.figure(figsize=(7,7))
plt.scatter(tf.squeeze(X), y, c="k", alpha=0.5)
plt.xlabel("Edad [años]"); plt.ylabel("Estatura [cm]")

#%% [markdown] 
# La regresión lineal es un modelo matemático que permite aproximar el valor de una variable independiente $y_i$ con un conjunto de variables independientes $\vec{x}_i=[x_1,x_2,\dots,x_m]^T$ y por medio de una relación líneal:
#
# $$
# \tilde{y_i}=w_1 x_1+w_2 x_2+\dots+w_m x_m +w_0=\vec{w}\cdot\vec{x}_i
# $$
#
# Generalmente, para poder estimar los parámetros del modelo $w_i$ se necesitan varios ejemplos u observaciones. Siguiendo la convención típica de *machine learning*, partimos de una matríz $\mathbf{X}$ de tamaño $N\times m$ (donde $N$ es el número de observaciones y $m$ es el número de variables independientes) y un vector de etiquetas $\vec{y}=[y_1,y_2,\dots,y_N]^T$.
#
# Veamos como implementar este modelo en Keras, el modelo de regresión lineal en keras se define utilizando el objeto ```tf.keras.models.Sequential``` (más adelante entraremos en detalle sobre los tipos de modelos en keras) y el bloque funcional ```tf.keras.layers.Dense```, el cual define unos pesos $w$ (```tf.Variable```) y las operaciones necesarias para realizar una predicción.

#%% 
# Definimos un modelo en keras
model = tf.keras.models.Sequential()
# Creamos una capa densa lineal (modelo lineal)
reg_lay = tf.keras.layers.Dense(units=1, activation="linear", input_shape=(1, ))
# Agregamos la capa lineal
model.add(reg_lay)
# Veamos un resumen del modelo
model.summary()

#%% [markdown] 
# En keras, el modelo lineal se compone de dos parámetros: los pesos ($w_1,w_2,\dots,w_m$) y el sesgo o intercepto ($w_0$):

#%% 
 # Extraemos los parámetros de la capa
params = reg_lay.weights
# pesos
print("Los pesos del modelo son:\n{}".format(params[0]))
# sesgo
print("El sesgo del modelo es:\n{}".format(params[1]))

#%% [markdown] 
# Inicialmente, Keras genera los pesos del modelo de forma aleatoria (los cuales se estimarán durante el entrenamiento), veamos un ejemplo de esto: 

#%% 
# Evaluamos el modelo (pesos aleatorios) en el conjunto de datos
y_tilde = model(X)

#Visualizamos las predicciones
plt.figure(figsize=(7, 7))
plt.scatter(tf.squeeze(X), y, c="k", alpha=0.5)
plt.plot(tf.squeeze(X), tf.squeeze(y_tilde), c="k", alpha=0.5)
plt.xlabel("Edad [años]"); plt.ylabel("Estatura [metros]")

#%% [markdown] 
# En el caso de la regresión lineal las etiquetas son valores continúos $y_i\in \mathbb{R}$, por lo cual, lo que se busca con el modelo lineal es aproximar de la mejor forma un valor predicho $\widetilde{y}_i$ al valor real $y_i$. Esto se puede ver como un problema de optimización en el que se busca que la distancia (error cuadrático medio) entre estos dos valores sea lo más pequeña posible, es decir, se optimiza la siguiente función de pérdida:
#
# $$
# \mathcal{L}(w)=\frac{1}{N}\sum_{i=1}^{N}(y_i-\widetilde{y}_i)^2
# $$
#
# Para entrenar el modelo debemos compilarlo, es decir, especificar un optimizador y una función de pérdida:

#%% 
# Optimizador
opt = tf.optimizers.Adam(lr=1e-1)
# compilación
model.compile(optimizer=opt, loss="mse")

#%% [markdown] 
# Finalmente, para entrenar el modelo utilizamos el método ```fit```:

#%%
# Entrenamos el modelo
model.fit(X, y, epochs=100)

#%% [markdown] 
# Veamos los pesos estimados:

#%% 
# Pesos
print("Los pesos del modelo son:\n{}".format(params[0]))
# sesgo
print("El sesgo del modelo es:\n{}".format(params[1]))


#%% [markdown] 
# Y las predicciones:

#%% 
# Evaluamos el modelo (pesos aleatorios) en el conjunto de datos
y_tilde = model.predict(X)

#Visualizamos las predicciones
plt.figure(figsize=(7, 7))
plt.scatter(tf.squeeze(X), y, c="k", alpha=0.5)
plt.plot(tf.squeeze(X), tf.squeeze(y_tilde), c="k", alpha=0.5)
plt.xlabel("Edad [años]"); plt.ylabel("Estatura [metros]")

#%% [markdown] 
# #### Regresión Logística 
#
# Otro modelo básico es la regresión logística, para mostrar su implementación en Keras utilizaremos un dataset sintético que generaremos utilizando la librería de Scikit-learn:
#

#%% 
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=[(-1, -1), (1, 1)], cluster_std=0.3)
X = tf.constant(X)
y = tf.constant(y)

# grafiquemos los datos
plt.figure(figsize=(7,7))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")

#%% [markdown] 
#
# La regresión logística es un modelo de clasificación binaria, es decir, las etiquetas pueden tomar el valor de 0 o 1. El modelo logístico se muestra a continuación:
#
# $$
# \tilde{y_i}=\frac{1}{1+e^{w_1 x_1+w_2 x_2+\dots+w_m x_m +w_0}}=\frac{1}{1+e^{\vec{w}\cdot\vec{x}_i}}
# $$
#
# Esto es equivalente a la siguiente expresión:
#
# $$
# \tilde{y_i}=\text{sigmoid}(\vec{w}\cdot\vec{x}_i)
# $$
#
#%% [markdown] 
#
# Veamos la implementación de este modelo en keras. Al igual que la regresión lineal, el modelo logístico en keras se define utilizando el objeto ```tf.keras.models.Sequential``` y el bloque funcional ```tf.keras.layers.Dense```, el cual define unos pesos $w$ (```tf.Variable```) y las operaciones necesarias para realizar una predicción.
#
# La regresión logística también se puede abordar como un problema de optimización con una función de pérdida conocida como entropía cruzada binaria, la cual se muestra a continuación:
#
# $$\mathcal{L}(w)=-\frac{1}{N}\sum[y_i\log(\tilde{y}_i)]+(1-y_i)\log(1-\tilde{y}_i)]$$
#
# Veamos la implementación de este modelo en Keras:

#%% 
# Definimos el modelo
model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                1,
                input_shape=(2, ),
                activation="sigmoid"
                )
            ]
        )

#%%
# Comiplamos el modelo
opt = tf.optimizers.Adam(lr=1e-2)
model.compile(optimizer=opt, loss="categorical_crossentropy")

#%% [markdown]
# Ahora, vamos a entrenar el modelo:

#%%
model.fit(X, y, epochs=100)

#%% [markdown] 
# Podemos visualizar las regiones de decisión del modelo entrenado:

#%% 
plot_decision_region(X, y, model)
