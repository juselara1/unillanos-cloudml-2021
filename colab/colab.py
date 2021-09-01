#%% [markdown]
## Taller de Machine Learning en la Nube con Python - Google Colaboratory
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
### ¿Qué es Google Colaboratory?
# Se trata de un servicio gratuito para computo en la nube que permite ejecutar código en Python desde cualquier navegador. Es un servicio que no requiere mucha configuración, ofrece procesamiento en CPU y GPU gratuito, acceso a RAM y almacenamiento en disco.
#
# Google Colaboratory esta pensado para ejecutar Jupyter Notebooks, los cuales son una herramienta que permite correr, visualizar y evaluar fragmentos de código ordenados en celdas, por ejemplo:

#%%
x = 1
print(x)

#%% [markdown]
# Desde Colaboratory podemos acceder a otros servicios en la nube, por ejemplo Google Drive:

#%%
from google.colab import drive
drive.mount("content/drive")

#%% [markdown]
# Veamos unas pruebas generales:

#%% [markdown] 
# * RAM

#%%
x = list(range(int(1e8)))

#%% [markdown]
# * Disco

#%%
with open("test.txt", "w") as f:
    for x_i in x:
        f.write(f"{x_i}\n")
