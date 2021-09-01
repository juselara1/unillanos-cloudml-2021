#%% [markdown]
# # Taller de Machine Learning en la Nube con Python - APIs de Machine Learning en la Nube
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
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#%% [markdown]
# ## Servicios en la Nube
# Existen diversos servicios para machine learning en la nube. Distintos proveedores como Google, Microsoft, Amazon, entre otros; dan acceso a APIs con distintos modelos ya pre-entrenados y listos para usar. En este caso utilizaremos APIs gratuitas que ofrece DeepAI.
#
# Para usar estos servicios debemos registrarnos en DeepAI y obtener un API key.
#%%
api_key = ""

#%% [markdown]
# ## Image Colorization
# Este modelo nos permite colorear imágenes en blanco y negro.

# Primero, cargamos la imagen en blanco y negro:
# **Puede descargar la imagen usando el siguiente comando**
#
# ```python
# !wget 
# ```

#%%
im_bw = cv2.imread("data/bw_image.jpg") 

#%% [markdown]
# Realizamos un post al endpoint correspondiente:

#%%
r = requests.post(
        "https://api.deepai.org/api/colorizer",
        files={"image": open("data/bw_image.jpg", "rb")},
        headers={"api-key": api_key}
        )
print(r.status_code)
print(r.json())

#%% [markdown]
# Realizamos un get para extraer la imagen resultante:

#%%
r2 = requests.get(
        r.json()["output_url"]
        )
im_color = np.frombuffer(r2.content, np.uint8)
im_color = cv2.imdecode(im_color, cv2.IMREAD_ANYCOLOR)

#%% [markdown]
# Finalmente, veamos los resultados:

#%%
fig, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].imshow(im_bw)
ax[0].axis("off")
ax[0].set_title("Original")
ax[1].imshow(im_color)
ax[1].axis("off")
ax[1].set_title("Resultado")

#%% [markdown]
# ## Facial Recognition
# Este servicio nos permite detectar rostros dentro de imágenes.
#
# Primero cargamos una imagen con rostros.
# 
# **Puede descargar la imagen usando el siguiente comando:**
#
# ```sh
# !wet 
# ```

#%%
im = cv2.imread("data/people.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#%%
r = requests.post(
        "https://api.deepai.org/api/facial-recognition",
        files={"image": open("data/people.png", "rb")},
        headers={"api-key": api_key}
        )
print(r.status_code)
print(r.json())

#%% [markdown]
# Veamos los resultados:

#%%
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.imshow(im)
for bbox in r.json()["output"]["faces"]:
    coords = bbox["bounding_box"]
    rect = Rectangle(
            coords[:2],
            coords[2],
            coords[3],
            linewidth=5, edgecolor='b',
            facecolor='none'
            )
    ax.add_patch(rect)
ax.axis("off")

#%% [markdown]
# ## Sentiment Analysis
# El servicio de análisis de sentimientos permite clasificar sentimientos en textos.

#%% [markdown]
# Cargamos los textos

#%%
with open("data/angry.txt") as f:
    angry = f.read()

with open("data/happy.txt") as f:
    happy = f.read()

#%% [markdown]
# Hacemos las dos peticiones:

#%%
# Texto angry.txt
r = requests.post(
        "https://api.deepai.org/api/sentiment-analysis",
        files={"text": open("data/angry.txt", "r")},
        headers={"api-key": api_key}
        )
print(r.status_code)
print(angry)
print(r.json())

#%%
# Texto happy.txt
r = requests.post(
        "https://api.deepai.org/api/sentiment-analysis",
        files={"text": open("data/happy.txt", "r")},
        headers={"api-key": api_key}
        )
print(r.status_code)
print(happy)
print(r.json())

#%% [markdown]
# ## Text Summarization
# Este servicio nos permite acortar textos.
#
# Primero, cargamos el texto:

#%%
with open("data/long_text.txt") as f:
    long_text = f.read()

#%% [markdown]
# Ahora, hacemos la petición:

#%%
r = requests.post(
        "https://api.deepai.org/api/summarization",
        files={"text": open("data/long_text.txt", "r")},
        headers={"api-key": api_key}
        )
print(r.status_code)
print(long_text)
print(r.json())

#%% [markdown]
# ## Toonify
# Con este servicio podemos caricaturizar imágenes.

#%%
im = cv2.imread("data/kubrick.jpg")
r = requests.post(
        "https://api.deepai.org/api/toonify",
        files={"image": open("data/kubrick.jpg", "rb")},
        headers={"api-key": api_key}
        )

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#%% 
r2 = requests.get(
        r.json()["output_url"]
        )
im_toon = np.frombuffer(r2.content, np.uint8)
im_toon = cv2.imdecode(im_toon, cv2.IMREAD_ANYCOLOR)
im_toon = cv2.cvtColor(im_toon, cv2.COLOR_BGR2RGB)

#%%
fig, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].imshow(im)
ax[0].axis("off")
ax[0].set_title("Original")
ax[1].imshow(im_toon)
ax[1].axis("off")
ax[1].set_title("Resultado")
fig.savefig("temp.png")
