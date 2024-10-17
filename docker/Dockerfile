# Usamos una imagen base oficial de Python, en este caso, Python 3.11 versión slim.
# La versión slim es más ligera, lo que significa que solo incluye los componentes esenciales.
FROM python:3.11-slim

# Establecemos el directorio de trabajo dentro del contenedor. Aquí es donde
# se ejecutarán los comandos a continuación y donde se guardarán los archivos de la aplicación.
WORKDIR /app

# Copiamos todos los archivos del directorio actual (desde el host) al directorio de trabajo en el contenedor.
# Esto permite que todo el código y archivos necesarios se muevan al contenedor.
ADD . /app

# Actualizamos el índice de paquetes del sistema operativo. Esto es necesario
# para asegurarnos de que los paquetes que instalemos estén actualizados.
RUN apt-get update

# Instalamos una biblioteca específica del sistema operativo llamada `libgomp1`.
# Esta biblioteca es necesaria para ejecutar operaciones paralelas con OpenMP,
# que puede ser requerida por algunas dependencias de Python.
RUN apt-get install libgomp1

# Instalamos las dependencias de Python listadas en el archivo `requirements.txt`.
# `pip` busca en este archivo todas las bibliotecas y versiones necesarias para la aplicación.
RUN pip install -r requirements.txt

# Exponemos el puerto 8000 para permitir que el tráfico entre y salga del contenedor a través de ese puerto.
# Esto es esencial porque la aplicación FastAPI se ejecuta en ese puerto.
EXPOSE 8000

# Definimos el comando predeterminado que se ejecutará cuando el contenedor arranque.
# En este caso, ejecuta el archivo `app.py` con Python, que es el punto de entrada de la aplicación FastAPI.
CMD [ "python", "app.py" ]
