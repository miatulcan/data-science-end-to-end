"""Ingeniería de características sobre una muestra limitada de ventas."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
ARCHIVO = CARPETA.parents[2] / 'data' / 'processed' / 'obt_sales.csv'
OBJETIVO = 'UNITS'
LIMITE_LECTURA = 100_000
TAMANO_MUESTRA = 20_000
SEMILLA = 42


def cargar_muestra():
    # nrows evita recorrer o cargar todo el archivo de 13 GB.
    bloque = pd.read_csv(ARCHIVO, nrows=LIMITE_LECTURA, low_memory=False)
    return bloque.sample(n=min(TAMANO_MUESTRA, len(bloque)), random_state=SEMILLA).copy()


def crear_features(datos):
    datos = datos.copy()
    # Hallazgo Semana 8: la ausencia de precio tiene un contexto particular.
    # Válida SOLO si el precio (o su ausencia) se conoce ANTES del día objetivo.
    datos['precio_no_disponible'] = datos['SELL_PRICE'].isna().astype(int)
    # Tienda y categoría están disponibles antes de las ventas.
    datos['tienda_categoria'] = datos['STORE_ID'] + '__' + datos['CAT_ID']
    # Ejercicio 1 adaptado: calendario diario, pues no existen horas.
    fecha = pd.to_datetime(datos['DATE'], errors='raise')
    datos['dia_semana'] = fecha.dt.dayofweek
    datos['es_fin_de_semana'] = datos['dia_semana'].ge(5).astype(int)
    datos['dia_sin'] = np.sin(2 * np.pi * datos['dia_semana'] / 7)
    datos['dia_cos'] = np.cos(2 * np.pi * datos['dia_semana'] / 7)
    return datos


def ejecutar():
    datos = crear_features(cargar_muestra())
    resumen = dict(archivo='data/processed/obt_sales.csv', objetivo=OBJETIVO,
                   limite_lectura=LIMITE_LECTURA, muestra_filas=len(datos), semilla=SEMILLA,
                   metodo='Aleatoria dentro de las primeras 100000 filas; no es muestra global',
                   fecha_min=str(datos['DATE'].min()), fecha_max=str(datos['DATE'].max()),
                   tiendas=sorted(datos['STORE_ID'].unique().tolist()))
    for columna in ['precio_no_disponible', 'tienda_categoria', 'es_fin_de_semana']:
        resumen[columna] = datos.groupby(columna)[OBJETIVO].agg(['count', 'mean', 'min', 'max']).reset_index().to_dict('records')
    (CARPETA / 'resultados_features.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(datos[['DATE', OBJETIVO, 'precio_no_disponible', 'tienda_categoria']].head().to_string(index=False))
    return datos, resumen


if __name__ == '__main__':
    ejecutar()
