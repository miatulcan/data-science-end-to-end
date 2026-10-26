"""Ejercicios 1 y 2 adaptados a la muestra del proyecto de ventas."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

CARPETA = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pipeline_proyecto', CARPETA.parent / 'plantilla-pipeline' / 'pipeline_proyecto.py')
proyecto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proyecto)


def ejecutar():
    # TODO 1 : división por fecha, porque se predice el futuro.
    datos, X_ent, X_prueba, y_ent, y_prueba, corte = proyecto.cargar_y_dividir()
    # TODO 2: tres grupos: imputación/escalado, imputación/one-hot, passthrough.
    preparacion = proyecto.construir_preparacion()
    # TODO 3: fit solo en entrenamiento; prueba únicamente se transforma.
    ent_listo = preparacion.fit_transform(X_ent)
    prueba_lista = preparacion.transform(X_prueba)
    nombres = preparacion.get_feature_names_out()
    # TODO 4: medianas aprendidas frente a entrenamiento y muestra completa.
    medianas = preparacion.named_transformers_['numericas'].named_steps['imputar'].statistics_
    assert np.allclose(medianas, X_ent[proyecto.NUMERICAS].median())
    # TODO 5: una combinación de tienda y categoría nunca vista.
    nuevo = X_prueba.iloc[[0]].copy()
    nuevo['tienda_categoria'] = 'TIENDA_NUEVA__FOODS'
    encoder = preparacion.named_transformers_['categoricas'].named_steps['codificar']
    assert 'TIENDA_NUEVA__FOODS' not in encoder.categories_[0]
    nuevo_listo = preparacion.transform(nuevo)
    mascara = np.array([n.startswith('categoricas__tienda_categoria_') for n in nombres])
    activas = int(np.count_nonzero(nuevo_listo[0, mascara]))
    assert activas == 0 and nuevo_listo.shape[1] == ent_listo.shape[1]
    # Comprobación artificial separada: la muestra no tiene faltantes en estas
    # entradas. No se alteran sus datos ni sus estadísticas para simularlos.
    con_faltantes = X_prueba.iloc[[0]].copy()
    for columna in proyecto.NUMERICAS + proyecto.CATEGORICAS:
        con_faltantes[columna] = np.nan
    faltantes_listo = preparacion.transform(con_faltantes)
    escalador = preparacion.named_transformers_['numericas'].named_steps['escalar']
    esperados = (medianas - escalador.mean_) / escalador.scale_
    assert np.isfinite(faltantes_listo).all()
    assert np.allclose(faltantes_listo[0, :len(proyecto.NUMERICAS)], esperados)
    # Contraste real con get_dummies aplicado independientemente.
    dummies_ent = pd.get_dummies(X_ent[proyecto.CATEGORICAS], dtype=int)
    dummies_nuevo = pd.get_dummies(nuevo[proyecto.CATEGORICAS], dtype=int)

    # Ejercicio 1: comparación didáctica; la variante con fuga NO se recomienda.
    # Para estabilidad numérica con intercepto, dejar una categoría de referencia.
    def preparacion_modelo():
        prep = proyecto.construir_preparacion()
        prep.set_params(categoricas__codificar__drop='first')
        return prep

    correcto = Pipeline([('preparar', preparacion_modelo()), ('modelo', LinearRegression())])
    correcto.fit(X_ent, y_ent)
    mae_correcto = mean_absolute_error(y_prueba, correcto.predict(X_prueba))
    # ERROR DELIBERADO: el fit conoce las filas futuras de la muestra.
    con_fuga = preparacion_modelo()
    con_fuga.fit(datos[proyecto.NUMERICAS + proyecto.CATEGORICAS + proyecto.YA_LISTAS])
    modelo_fuga = LinearRegression().fit(con_fuga.transform(X_ent), y_ent)
    mae_fuga = mean_absolute_error(y_prueba, modelo_fuga.predict(con_fuga.transform(X_prueba)))
    resumen = dict(
        muestra_filas=len(datos), fecha_corte=corte.date().isoformat(),
        forma_entrenamiento=list(ent_listo.shape), forma_prueba=list(prueba_lista.shape),
        columnas_salida=nombres.tolist(),
        grupos=dict(numericas=len(proyecto.NUMERICAS), categoricas=int(mascara.sum()), ya_listas=len(proyecto.YA_LISTAS)),
        medianas_aprendidas=dict(zip(proyecto.NUMERICAS, medianas.tolist())),
        medianas_entrenamiento=X_ent[proyecto.NUMERICAS].median().to_dict(),
        medianas_toda_muestra=datos[proyecto.NUMERICAS].median().to_dict(),
        categoria_nueva='TIENDA_NUEVA__FOODS', forma_nuevo=list(nuevo_listo.shape),
        columnas_categoricas_activas_nuevo=activas,
        columnas_dummies_entrenamiento=len(dummies_ent.columns),
        columnas_dummies_nuevo=dummies_nuevo.columns.tolist(),
        esquema_dummies_coincide=dummies_ent.columns.equals(dummies_nuevo.columns),
        prueba_faltantes_artificiales='Transformación finita; se usaron las medianas de entrenamiento',
        mae_correcto=float(mae_correcto), mae_con_fuga_didactica=float(mae_fuga),
        diferencia_mae=float(mae_correcto - mae_fuga),
        nota='Comparación didáctica; MAE sobre muestra sesgada, no evaluación definitiva del proyecto',
    )
    (CARPETA / 'resultados_ejercicios.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return resumen


if __name__ == '__main__':
    ejecutar()
