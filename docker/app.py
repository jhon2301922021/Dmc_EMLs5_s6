import pandas as pd  # Importamos la biblioteca pandas para manejar y crear dataframes.
import mlflow  # Importamos mlflow para cargar modelos previamente entrenados.
from fastapi import FastAPI  # Importamos FastAPI para crear la API.
import uvicorn  # Importamos uvicorn para correr la aplicación de FastAPI.
from pydantic import BaseModel  # Importamos BaseModel para definir la estructura de los datos que vamos a recibir.

# Inicializamos la aplicación FastAPI
app = FastAPI()

# Cargamos el modelo de mlflow
# Aquí configuramos la URI de tracking de MLflow que apunta al endpoint de una instancia EC2.
mlflow.set_tracking_uri("http://ec2_instance_endpoint:5000")
# Cargamos el modelo de producción de mlflow que se ha registrado bajo el nombre 'rent_model'.
model = mlflow.pyfunc.load_model("models:/rent_model/Production")

# Definimos un modelo de datos usando Pydantic. Este esquema define qué información esperamos recibir en el endpoint.
class Data(BaseModel):
    BHK: int = 2  # Número de habitaciones. Por defecto, 2.
    Size: int = 2000  # Tamaño en pies cuadrados o metros cuadrados. Por defecto, 2000.
    Area_Type: str = 'Super Area'  # Tipo de área (e.g., Super Area). Por defecto, 'Super Area'.
    City: str = 'Hyderabad'  # Ciudad. Por defecto, 'Hyderabad'.
    Furnishing_Status: str = 'Unfurnished'  # Estado del mobiliario: Amueblado o no. Por defecto, 'Unfurnished'.
    Tenant_Preferred: str = 'Bachelors'  # Tipo de inquilino preferido (e.g., Solteros). Por defecto, 'Bachelors'.
    Bathroom: int = 3  # Número de baños. Por defecto, 3.
    Point_of_Contact: str = 'Contact Owner'  # Persona de contacto. Por defecto, 'Contact Owner'.

# Función que toma los datos enviados en formato JSON y los convierte en un DataFrame.
def create_dataframe(data: Data):
    # Creamos un DataFrame a partir de los datos, usando un diccionario para el mapeo.
    df = pd.DataFrame(dict(data), index=[0])  # Asignamos índice 0 al DataFrame.

    # Renombramos las columnas del DataFrame para que coincidan con los nombres que espera el modelo.
    df = df.rename(columns={'BHK': 'BHK',
                            'Size': 'Size',
                            'Area_Type': 'Area Type',
                            'City': 'City',
                            'Furnishing_Status': 'Furnishing Status',
                            'Tenant_Preferred': 'Tenant Preferred',
                            'Bathroom': 'Bathroom',
                            'Point_of_Contact': 'Point of Contact'})
    return df

# Definimos el endpoint POST '/predict' que recibe los datos y devuelve una predicción.
@app.post("/predict")
def predict(data: Data):
    # Convertimos los datos recibidos en un DataFrame usando la función anterior.
    data = create_dataframe(data)

    # Usamos el modelo cargado para hacer una predicción.
    prediction = model.predict(data)

    # Retornamos la predicción como un diccionario JSON.
    return {'prediction': prediction[0]}

# Punto de entrada de la aplicación. Aquí especificamos que el servidor se ejecutará en el puerto 8000.
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
