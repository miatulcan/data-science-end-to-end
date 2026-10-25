import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder

# Busca el CSV junto al script, sin depender de la carpeta de la terminal.
viajes = pd.read_csv(Path(__file__).parent / "viajes_taxi.csv")

viajes_clean = viajes.query("tarifa < 999.99 and duracion_min > 0").copy()

print(viajes_clean.head())
print(viajes_clean.info())

fecha_inicio = pd.to_datetime(viajes_clean['fecha_hora_inicio'])

viajes_clean['fecha_hora_inicio'] = fecha_inicio
viajes_clean['hora'] = fecha_inicio.dt.hour
viajes_clean['dia_semana'] = fecha_inicio.dt.dayofweek

print(viajes_clean.head())

viajes_clean['hora_sin'] = np.sin(2 * np.pi * viajes_clean['hora'] / 24)
viajes_clean['hora_cos'] = np.cos(2 * np.pi * viajes_clean['hora'] / 24)

print(viajes_clean[['fecha_hora_inicio', 'hora', 'hora_sin', 'hora_cos']].head())

#revisar qué es one hot y qué es odinal
#no se recomienda usar get dummies. 
#onehot encoder aprende de las categprías en fit y las recuerda.
#se recomienda usar comulnTransformer 
#cuando hay data temporal no se puede hacer una parte de prueba y otra de test de manera random, debe existir una fecha de corte.
viajes_codificado = pd.get_dummies(viajes_clean, columns = ['zona_origen', 'zona_destino'])
print(viajes_codificado.head())

#creacion de variables ordinales: usando una tuberia
print(viajes_codificado)

encoder = OrdinalEncoder()

#fir_transform no regresa una columna limpia
viajes_codificado['metodo_pago_ordinal'] = encoder.fit_transform(viajes_codificado[['metodo_pago']]).ravel()
print(viajes_codificado.head())

#una regresion lineal con un monton de columnas no es optimo
#regression_model = LinearRegression()
