"""Modelos iniciales del proyecto de tarifas de viajes.

Genera una línea base, un modelo lineal regularizado y un ensamblaje para
comparar su error en entrenamiento y en prueba con la misma preparación y la
misma división 80/20.
"""

import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

viajes = pd.read_csv("viajes_features.csv")

NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]

X = viajes[NUMERICAS + YA_LISTAS + CATEGORICAS]
y = viajes["tarifa"]
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)


def preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
    ])


resultados = []


def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio

    mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))

    resultados.append({
        "modelo": nombre,
        "mae_entrenamiento": round(mae_ent, 4),
        "mae_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "segundos": round(segundos, 3),
    })
    print(f"{nombre:20s} entrenamiento {mae_ent:8.4f}   prueba {mae_prueba:8.4f}   tiempo {segundos:7.3f}s")
    return pipeline


# Línea base: predicción constante con la mediana de la tarifa de entrenamiento.
linea_base = DummyRegressor(strategy="median")
evaluar("linea base", linea_base)

# Modelo lineal regularizado para evitar sobreajuste.
# Ridge agrega penalización L2 sobre los coeficientes del modelo.
lineal_regularizada = Ridge(alpha=10.0)
evaluar("ridge alpha 10", lineal_regularizada)

# Ensamblaje por boosting: combina varios árboles para mejorar la generalización.
ensamblaje = GradientBoostingRegressor(random_state=42)
evaluar("gradient boosting", ensamblaje)

# Guardar la comparación final para el entregable 2.
tabla = pd.DataFrame(resultados)
tabla.to_csv("resultados_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))
