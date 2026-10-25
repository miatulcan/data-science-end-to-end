"""Ejercicio 2 adaptado: tiendas de ventas en lugar de zonas de taxis."""
from pathlib import Path
import importlib.util
import json
import pandas as pd

CARPETA = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('features_proyecto', CARPETA.parent / 'plantilla-features' / 'features_proyecto.py')
proyecto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proyecto)


def disponibilidad(columna):
    if columna == 'UNITS':
        return 'NO: objetivo conocido al terminar el día'
    if columna == 'REVENUE':
        return 'NO: depende de las unidades vendidas ese día'
    if columna in {'IS_SALES_SPIKE', 'IS_SUSPECTED_STOCKOUT_CONSERVATIVE', 'IS_SUSPECTED_STOCKOUT_NB'}:
        return 'NO: diagnóstico con ventas observadas; excluir'
    if columna in {'SELL_PRICE', 'IS_PRE_LAUNCH', 'IS_STORE_CLOSED'}:
        return 'CONDICIONAL: verificar que procede de un plan anterior al día objetivo'
    if columna == '_LOADED_AT':
        return 'NO USAR: metadato de carga, no predictor de demanda'
    if columna.startswith('EVENT_') or columna in {'IS_EVENT_DAY', 'N_EVENTS', 'IS_CHRISTMAS_CLOSED', 'IS_SNAP'}:
        return 'CONDICIONAL: calendario o programa publicado antes del día objetivo'
    return 'SÍ: identificador o calendario conocido antes del día objetivo'


def ejecutar():
    originales = proyecto.cargar_muestra()
    datos = proyecto.crear_features(originales)
    # 1. Indicador de contexto en lugar de tarifa fija.
    seleccion = datos.loc[datos['precio_no_disponible'].eq(1), 'UNITS']
    # 2. One-hot de tienda: una columna por categoría, sin orden artificial.
    one_hot = pd.get_dummies(datos['STORE_ID'], prefix='tienda', dtype=int)
    # 3. Ordinal: solo para comparar; no usar como orden real de las tiendas.
    mapa = {tienda: i for i, tienda in enumerate(sorted(datos['STORE_ID'].unique()))}
    datos['tienda_ordinal'] = datos['STORE_ID'].map(mapa)
    # 4. Revisar todas las columnas originales.
    resumen = dict(muestra_filas=len(datos),
                   precio_ausente=dict(filas=int(seleccion.count()),
                                       unidades_min=float(seleccion.min()) if len(seleccion) else None,
                                       unidades_max=float(seleccion.max()) if len(seleccion) else None),
                   one_hot_columnas=one_hot.columns.tolist(), ordinal_columnas=1,
                   mapa_ordinal=mapa,
                   disponibilidad={col: disponibilidad(col) for col in originales.columns})
    (CARPETA / 'resultados_codificacion.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return datos, one_hot, resumen


if __name__ == '__main__':
    ejecutar()
