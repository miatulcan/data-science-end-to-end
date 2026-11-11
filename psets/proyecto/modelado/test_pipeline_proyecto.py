"""Pruebas del pipeline de pronostico de ventas."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ_REPO = Path(__file__).resolve().parents[3]
RUTA_PIPELINE = (
    RAIZ_REPO
    / "laboratorios"
    / "semana-09-sesion-02"
    / "plantilla-pipeline"
    / "pipeline_proyecto.py"
)

spec = importlib.util.spec_from_file_location("pipeline_proyecto", RUTA_PIPELINE)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo cargar el pipeline del proyecto: {RUTA_PIPELINE}")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_objetivo_no_esta_entre_las_features():
    columnas_modelo = (
        pipeline.NUMERICAS + pipeline.CATEGORICAS + pipeline.YA_LISTAS
    )
    assert pipeline.OBJETIVO == "UNITS"
    assert pipeline.OBJETIVO not in columnas_modelo


def test_features_de_calendario_y_tienda_categoria():
    datos = pd.DataFrame(
        {
            "DATE": ["2016-04-02", "2016-04-04"],
            "STORE_ID": ["CA_1", "WI_2"],
            "CAT_ID": ["FOODS", "HOUSEHOLD"],
            "SELL_PRICE": [2.5, np.nan],
        }
    )

    preparados = pipeline.features.crear_features(datos)

    assert preparados["tienda_categoria"].tolist() == [
        "CA_1__FOODS",
        "WI_2__HOUSEHOLD",
    ]
    assert preparados["es_fin_de_semana"].tolist() == [1, 1]
    assert np.isclose(preparados.loc[0, "dia_sin"], np.sin(2 * np.pi * 5 / 7))
    assert np.isclose(preparados.loc[0, "dia_cos"], np.cos(2 * np.pi * 5 / 7))


def test_preparacion_acepta_categoria_no_vista():
    entrenamiento = pd.DataFrame(
        {
            "YEAR": [2015, 2016],
            "D_NUM": [100, 200],
            "tienda_categoria": ["CA_1__FOODS", "CA_2__HOBBIES"],
            "es_fin_de_semana": [0, 1],
            "dia_sin": [0.0, 0.5],
            "dia_cos": [1.0, 0.5],
        }
    )
    nuevo = pd.DataFrame(
        {
            "YEAR": [2016],
            "D_NUM": [201],
            "tienda_categoria": ["WI_3__HOUSEHOLD"],
            "es_fin_de_semana": [0],
            "dia_sin": [0.0],
            "dia_cos": [1.0],
        }
    )
    preparacion = pipeline.construir_preparacion()
    matriz_entrenamiento = preparacion.fit_transform(entrenamiento)
    matriz_nueva = preparacion.transform(nuevo)

    assert matriz_nueva.shape == (1, matriz_entrenamiento.shape[1])
    assert np.isfinite(matriz_nueva).all()
