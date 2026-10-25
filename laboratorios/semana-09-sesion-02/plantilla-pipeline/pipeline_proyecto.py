"""Preparación reproducible de una muestra de ventas, con división temporal."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CARPETA = Path(__file__).resolve().parent
RUTA_FEATURES = CARPETA.parents[1] / 'semana-09-sesion-01' / 'plantilla-features' / 'features_proyecto.py'
spec = importlib.util.spec_from_file_location('features_sesion_01', RUTA_FEATURES)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

OBJETIVO = 'UNITS'
NUMERICAS = ['YEAR', 'D_NUM']
CATEGORICAS = ['tienda_categoria']
YA_LISTAS = ['es_fin_de_semana', 'dia_sin', 'dia_cos']
# SELL_PRICE y precio_no_disponible quedan fuera: no se ha confirmado que
# procedan de precios planificados disponibles antes del día pronosticado.
# Se usa la alternativa de calendario descrita en la sesión 1.


def cargar_y_dividir():
    datos = features.crear_features(features.cargar_muestra())
    fechas = pd.to_datetime(datos['DATE'])
    fechas_unicas = sorted(fechas.unique())
    if len(fechas_unicas) < 2:
        raise ValueError('Se necesitan al menos dos fechas distintas para dividir.')
    # Cerca del 80% de fechas para entrenar; no se garantiza 80% de filas.
    posicion = max(1, min(int(len(fechas_unicas) * 0.8), len(fechas_unicas) - 1))
    corte = pd.Timestamp(fechas_unicas[posicion])
    es_ent = fechas.lt(corte)
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
    y = datos[OBJETIVO]
    return datos, X.loc[es_ent], X.loc[~es_ent], y.loc[es_ent], y.loc[~es_ent], corte


def construir_preparacion():
    return ColumnTransformer([
        ('numericas', Pipeline([
            ('imputar', SimpleImputer(strategy='median')),
            ('escalar', StandardScaler()),
        ]), NUMERICAS),
        ('categoricas', Pipeline([
            ('imputar', SimpleImputer(strategy='most_frequent')),
            ('codificar', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAS),
        ('ya_listas', 'passthrough', YA_LISTAS),
    ], remainder='drop')


def ejecutar():
    datos, X_ent, X_prueba, y_ent, y_prueba, corte = cargar_y_dividir()
    preparacion = construir_preparacion()
    # Único ajuste: se aprende exclusivamente de las filas de entrenamiento.
    X_ent_listo = preparacion.fit_transform(X_ent)
    X_prueba_listo = preparacion.transform(X_prueba)
    nombres = preparacion.get_feature_names_out().tolist()
    imputador = preparacion.named_transformers_['numericas'].named_steps['imputar']
    escalador = preparacion.named_transformers_['numericas'].named_steps['escalar']
    # Comprobaciones de fuga, esquema estable y datos aptos para el modelo.
    assert np.allclose(imputador.statistics_, X_ent[NUMERICAS].median())
    assert np.allclose(escalador.mean_, X_ent[NUMERICAS].mean())
    assert X_ent_listo.shape[1] == X_prueba_listo.shape[1] == len(nombres)
    assert np.isfinite(X_ent_listo).all() and np.isfinite(X_prueba_listo).all()
    assert pd.to_datetime(datos.loc[X_ent.index, 'DATE']).max() < pd.to_datetime(datos.loc[X_prueba.index, 'DATE']).min()
    resumen = dict(
        sklearn_version=sklearn.__version__, objetivo=OBJETIVO,
        muestra_filas=len(datos), limite_lectura=features.LIMITE_LECTURA,
        semilla=features.SEMILLA, fechas_distintas=int(datos['DATE'].nunique()),
        fecha_corte=corte.date().isoformat(),
        entrenamiento_hasta=str(datos.loc[X_ent.index, 'DATE'].max()),
        prueba_desde=str(datos.loc[X_prueba.index, 'DATE'].min()),
        forma_entrenamiento=list(X_ent_listo.shape), forma_prueba=list(X_prueba_listo.shape),
        numericas=NUMERICAS, categoricas=CATEGORICAS, ya_listas=YA_LISTAS,
        columnas_salida=nombres,
        faltantes_entrenamiento=X_ent.isna().sum().astype(int).to_dict(),
        medianas_aprendidas=dict(zip(NUMERICAS, imputador.statistics_.tolist())),
        medianas_entrenamiento=X_ent[NUMERICAS].median().to_dict(),
        medianas_toda_muestra=datos[NUMERICAS].median().to_dict(),
        medias_escalador=dict(zip(NUMERICAS, escalador.mean_.tolist())),
        medias_entrenamiento=X_ent[NUMERICAS].mean().to_dict(),
        medias_toda_muestra=datos[NUMERICAS].mean().to_dict(),
        comprobaciones='Medianas y medias de entrenamiento; fechas separadas; mismo esquema; matrices finitas',
    )
    (CARPETA / 'resultados_pipeline.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return preparacion, X_ent, X_prueba, y_ent, y_prueba, resumen


if __name__ == '__main__':
    ejecutar()
