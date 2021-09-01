#%% [markdown]
# # Taller de Machine Learning en la Nube con Python - Introducción a REST
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
# ## ¿Qué es REST?
#
# Represantational Siate transfer (REST), se trata de un estilo de diseño de arquitectura de software que fue creado para diseñar y desarrollar en la World Wide Web (www). REST ha sido usado en la industria para crear web APIs escalables, seguras y confiables. 
#
# Cuando una API sigue las restricciones REST se le conoce como RESTful, estas APIs generalmente utilizan métodos HTTP para acceder a recursos por medio de parámetros codificados en parámetros de URLs, JSON o XML.
#<img src="https://phpenthusiast.com/theme/assets/images/blog/what_is_rest_api.png" width="500">

#%% [markdown]
# ## Propiedades de REST
# REST tiene algunas propiedades generales:
# 
# * **Protocolo cliente/servidor**: cada mensaje HTTP contiene toda la información necesaria para completar una petición. Es decir, no hay necesidad de guardar información previa o mensajes anteriores.
# * **Operaciones bien definidas**: se utilizan una serie de operaciones como GET, POST, PUT y DELETE.
# * **Sintaxis universal**: esta sintaxis se utiliza para identificar los recursos, cada recurso se accede por medio de su URI.
# * **Uso de hipermedios**: la representación de la aplicación y de las transacciones se da típicamente en formato HTML o XML. Esto permite navegar recursos siguiendo enlaces sin requerir registros o elementos adicionales.
#
# En Python, existen diversas librerías para utilizar REST. Una de las mas usadas es `requests`

# **Nota**: para correr este notebook, debe ejecutar antes el código `main.py`.

#%%
import requests, cv2
import matplotlib.pyplot as plt

#%%
#%% [markdown]
# Veamos como enviar una imagen como un `post` al api ejecutando en local con `flask`.

#%%
im_original = cv2.imread("data/image.jpg")
files = {"image": open("data/image.jpg", "rb")}
r = requests.post(
        "http://localhost:5000/predict",
        files=files
        )
print(r.status_code)

#%% [markdown]
# El resultado viene en formato binario, realicemos la decodificación:

#%%
arr = np.frombuffer(r.content, np.uint8)
im_res = cv2.imdecode(arr, cv2.IMREAD_ANYCOLOR)

#%% [markdown]
# Finalmente, comparemos las dos imágenes:

#%%
fig, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].imshow(im_original)
ax[0].axis("off")
ax[0].set_title("Original")
ax[1].imshow(im_res)
ax[1].axis("off")
ax[1].set_title("Resultado")
