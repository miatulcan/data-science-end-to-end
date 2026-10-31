"""Compara el gradient boosting, la red neuronal y un LLM para tarifas de taxi.

Corre desde esta carpeta:

    python3 comparar_proyecto.py

El script entrena los modelos, escribe un prompt para 10 viajes de prueba
y guarda las metricas en resultados_proyecto.csv. Si existe
respuestas_llm_proyecto.txt con 10 numeros (uno por linea), tambien calcula
el MAE del LLM sobre esos mismos viajes.
"""

import math
import os
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARCHIVO_RESPUESTAS = "respuestas_llm_proyecto.txt"
PRECIO_POR_MILLON_DE_TOKENS = 0.15  # Referencia GPT-4o mini API, entrada estandar.

NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]
LEGIBLES = [
    "zona_origen", "zona_destino", "distancia_km", "pasajeros",
    "hora", "es_fin_de_semana", "metodo_pago",
]

viajes = pd.read_csv("viajes_features.csv")
viajes["hora"] = pd.to_datetime(viajes["fecha_hora_inicio"]).dt.hour
X = viajes[NUMERICAS + YA_LISTAS + CATEGORICAS + ["hora"]]
y = viajes["tarifa"]
X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42
)

muestra = X_prueba.sample(10, random_state=7)
y_muestra = y_prueba.loc[muestra.index]
ejemplos = X_ent.sample(30, random_state=7)
y_ejemplos = y_ent.loc[ejemplos.index]


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


def entrenar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    predicciones_prueba = pipeline.predict(X_prueba)
    predicciones_muestra = pipeline.predict(muestra)
    parametros = None
    modelo_entrenado = pipeline.named_steps["modelo"]
    if isinstance(modelo_entrenado, MLPRegressor):
        parametros = sum(coef.size for coef in modelo_entrenado.coefs_)
        parametros += sum(intercepto.size for intercepto in modelo_entrenado.intercepts_)

    resultado = {
        "modelo": nombre,
        "mae_prueba_completa": mean_absolute_error(y_prueba, predicciones_prueba),
        "mae_misma_muestra_10": mean_absolute_error(y_muestra, predicciones_muestra),
        "segundos_entrenamiento": segundos,
        "parametros_red": parametros,
        "costo_api_10000_viajes_usd": None,
    }
    return pipeline, resultado


def escribir_prompt():
    filas_ejemplo = ejemplos[LEGIBLES].copy()
    filas_ejemplo["tarifa"] = y_ejemplos.values
    texto = (
        "Estimas tarifas de taxi en dolares. Usa los viajes con tarifa como "
        "ejemplos. Para los 10 viajes sin tarifa, responde solo con 10 numeros, "
        "uno por linea y en el mismo orden. La distancia esta en kilometros; "
        "si pasajeros esta vacio, el dato falta.\n\n"
        "Viajes con tarifa:\n"
        + filas_ejemplo.to_csv(index=False)
        + "\nViajes sin tarifa:\n"
        + muestra[LEGIBLES].to_csv(index=False)
    )
    with open("prompt_llm_proyecto.txt", "w") as archivo:
        archivo.write(texto)
    return texto


def leer_respuestas_llm():
    with open(ARCHIVO_RESPUESTAS, "r") as archivo:
        lineas = [linea.strip() for linea in archivo if linea.strip()]
    if len(lineas) != len(muestra):
        raise ValueError(
            f"{ARCHIVO_RESPUESTAS} debe tener {len(muestra)} numeros; "
            f"se encontraron {len(lineas)}."
        )
    try:
        predicciones = [float(linea) for linea in lineas]
    except ValueError as error:
        raise ValueError(
            f"{ARCHIVO_RESPUESTAS} debe tener solo un numero por linea."
        ) from error
    if not all(math.isfinite(prediccion) for prediccion in predicciones):
        raise ValueError("Las respuestas del LLM deben ser numeros finitos.")
    return predicciones


gradient_boosting, resultado_gb = entrenar(
    "Gradient boosting", GradientBoostingRegressor(random_state=42)
)
red_neuronal, resultado_red = entrenar(
    "Red neuronal", MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
)
resultados = [resultado_gb, resultado_red]
prompt = escribir_prompt()

if os.path.exists(ARCHIVO_RESPUESTAS):
    predicciones_llm = leer_respuestas_llm()
    tokens_por_prompt = len(prompt) / 4
    costo_mensual = (
        tokens_por_prompt * (10_000 / len(muestra)) / 1_000_000
        * PRECIO_POR_MILLON_DE_TOKENS
    )
    resultados.append({
        "modelo": "LLM (respuestas_llm_proyecto.txt)",
        "mae_prueba_completa": None,
        "mae_misma_muestra_10": mean_absolute_error(y_muestra, predicciones_llm),
        "segundos_entrenamiento": None,
        "parametros_red": None,
        "costo_api_10000_viajes_usd": costo_mensual,
    })
else:
    print(
        f"Guarda las 10 respuestas del LLM, una por linea, en {ARCHIVO_RESPUESTAS} "
        "y vuelve a correr para completar su metrica."
    )

tabla = pd.DataFrame(resultados)
tabla.to_csv("resultados_proyecto.csv", index=False)
print(f"Prompt de 10 viajes escrito en prompt_llm_proyecto.txt ({len(prompt)} caracteres).")
print(tabla.round(4).to_string(index=False))
