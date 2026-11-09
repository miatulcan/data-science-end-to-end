"""Compara una línea base, Ridge y Gradient Boosting para ventas."""

import importlib.util
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline


CARPETA = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA.parents[2]
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
pipeline_proyecto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_proyecto)


def main() -> pd.DataFrame:
    datos, X_ent, X_prueba, y_ent, y_prueba, fecha_corte = (
        pipeline_proyecto.cargar_y_dividir()
    )
    modelos = [
        ("linea base", DummyRegressor(strategy="median")),
        ("ridge alpha 10", Ridge(alpha=10.0)),
        ("gradient boosting", GradientBoostingRegressor(random_state=42)),
    ]
    filas = []

    for nombre, modelo in modelos:
        pipeline = Pipeline(
            [
                ("preparar", pipeline_proyecto.construir_preparacion()),
                ("modelo", modelo),
            ]
        )
        inicio = time.perf_counter()
        pipeline.fit(X_ent, y_ent)
        segundos = time.perf_counter() - inicio

        mae_entrenamiento = mean_absolute_error(y_ent, pipeline.predict(X_ent))
        mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
        filas.append(
            {
                "modelo": nombre,
                "mae_entrenamiento": mae_entrenamiento,
                "mae_prueba": mae_prueba,
                "brecha": mae_prueba - mae_entrenamiento,
                "segundos": segundos,
                "semilla_muestra": pipeline_proyecto.features.SEMILLA,
                "fecha_corte": fecha_corte.date().isoformat(),
                "filas_entrenamiento": len(X_ent),
                "filas_prueba": len(X_prueba),
            }
        )

    resultados = pd.DataFrame(filas)
    destino = CARPETA / "resultados_proyecto.csv"
    resultados.to_csv(destino, index=False)
    print(resultados.to_string(index=False))
    print(f"\nResultados guardados en: {destino}")
    print(f"Filas de la muestra: {len(datos)}")
    return resultados


if __name__ == "__main__":
    main()
