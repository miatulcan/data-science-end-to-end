"""Compara el ensamblaje, una red neuronal y un LLM para ventas."""

import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
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
ARCHIVO_RESPUESTAS = CARPETA / "respuestas_llm_proyecto.txt"

spec = importlib.util.spec_from_file_location("pipeline_proyecto", RUTA_PIPELINE)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo cargar el pipeline del proyecto: {RUTA_PIPELINE}")
pipeline_proyecto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_proyecto)


def leer_respuestas_llm(ruta: Path, cantidad: int) -> list[float]:
    lineas = [linea.strip() for linea in ruta.read_text(encoding="utf-8").splitlines()]
    lineas = [linea for linea in lineas if linea]
    if len(lineas) != cantidad:
        raise ValueError(
            f"{ruta.name} debe contener {cantidad} números; "
            f"se encontraron {len(lineas)}."
        )
    try:
        predicciones = [float(linea) for linea in lineas]
    except ValueError as error:
        raise ValueError(f"{ruta.name} debe tener un número por línea.") from error
    if not all(math.isfinite(valor) for valor in predicciones):
        raise ValueError("Las predicciones del LLM deben ser números finitos.")
    return predicciones


def escribir_prompt(
    datos: pd.DataFrame,
    X_ent: pd.DataFrame,
    X_prueba: pd.DataFrame,
    y_ent: pd.Series,
    indices_ejemplos: pd.Index,
    indices_objetivo: pd.Index,
) -> Path:
    columnas = pipeline_proyecto.NUMERICAS + pipeline_proyecto.CATEGORICAS
    columnas += pipeline_proyecto.YA_LISTAS
    ejemplos = X_ent.loc[indices_ejemplos, columnas].copy()
    ejemplos.insert(0, "DATE", datos.loc[indices_ejemplos, "DATE"].astype(str))
    ejemplos["UNITS"] = y_ent.loc[indices_ejemplos].to_numpy()
    objetivo = X_prueba.loc[indices_objetivo, columnas].copy()
    objetivo.insert(0, "DATE", datos.loc[indices_objetivo, "DATE"].astype(str))
    prompt = (
        "Predice UNITS, las unidades vendidas en una combinación de fecha, "
        "tienda y categoría. Usa los ejemplos etiquetados solo como contexto; "
        "no inventes datos adicionales. Devuelve exactamente 10 números, "
        "uno por línea, en el mismo orden de los casos a predecir. Los valores "
        "pueden ser decimales no negativos.\n\n"
        "Ejemplos de entrenamiento:\n"
        f"{ejemplos.to_csv(index=False)}"
        "\nCasos de prueba sin etiqueta:\n"
        f"{objetivo.to_csv(index=False)}"
    )
    destino = CARPETA / "prompt_llm_proyecto.txt"
    destino.write_text(prompt, encoding="utf-8")
    return destino


def main() -> pd.DataFrame:
    datos, X_ent, X_prueba, y_ent, y_prueba, fecha_corte = (
        pipeline_proyecto.cargar_y_dividir()
    )
    indices_ejemplos = X_ent.sample(n=min(30, len(X_ent)), random_state=7).index
    muestra = X_prueba.sample(n=min(10, len(X_prueba)), random_state=7)
    y_muestra = y_prueba.loc[muestra.index]
    prompt_path = escribir_prompt(
        datos,
        X_ent,
        X_prueba,
        y_ent,
        indices_ejemplos,
        muestra.index,
    )

    modelos = [
        ("gradient boosting", GradientBoostingRegressor(random_state=42)),
        (
            "red neuronal",
            MLPRegressor(
                hidden_layer_sizes=(64,),
                max_iter=1000,
                random_state=42,
            ),
        ),
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
        predicciones_prueba = pipeline.predict(X_prueba)
        predicciones_muestra = pipeline.predict(muestra)
        parametros_red = None
        if isinstance(modelo, MLPRegressor):
            parametros_red = sum(coef.size for coef in pipeline.named_steps["modelo"].coefs_)
            parametros_red += sum(
                intercepto.size
                for intercepto in pipeline.named_steps["modelo"].intercepts_
            )
        filas.append(
            {
                "modelo": nombre,
                "mae_prueba_completa": mean_absolute_error(
                    y_prueba, predicciones_prueba
                ),
                "mae_misma_muestra_10": mean_absolute_error(
                    y_muestra, predicciones_muestra
                ),
                "segundos_entrenamiento": segundos,
                "parametros_red": parametros_red,
                "fecha_corte": fecha_corte.date().isoformat(),
                "filas_entrenamiento": len(X_ent),
                "filas_prueba": len(X_prueba),
            }
        )

    if ARCHIVO_RESPUESTAS.exists():
        predicciones_llm = leer_respuestas_llm(ARCHIVO_RESPUESTAS, len(muestra))
        filas.append(
            {
                "modelo": "LLM",
                "mae_prueba_completa": np.nan,
                "mae_misma_muestra_10": mean_absolute_error(
                    y_muestra, predicciones_llm
                ),
                "segundos_entrenamiento": np.nan,
                "parametros_red": np.nan,
                "fuente_predicciones": "asistente de esta conversación; sin API externa",
                "fecha_corte": fecha_corte.date().isoformat(),
                "filas_entrenamiento": len(X_ent),
                "filas_prueba": len(X_prueba),
            }
        )
    else:
        print(
            f"Falta {ARCHIVO_RESPUESTAS.name}: no se inventará la métrica del LLM. "
            "Pega allí las 10 respuestas al prompt, una por línea, y vuelve a correr."
        )

    resultados = pd.DataFrame(filas)
    destino = CARPETA / "resultados_proyecto.csv"
    resultados.to_csv(destino, index=False)
    print(resultados.to_string(index=False))
    print(f"\nPrompt guardado en: {prompt_path}")
    print(f"Resultados guardados en: {destino}")
    return resultados


if __name__ == "__main__":
    main()
