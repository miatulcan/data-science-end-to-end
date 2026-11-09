# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

El artículo distingue tres capas que conviene versionar por separado:

- El código del proyecto: se versiona con Git.
- Los datos de entrada y los datasets usados para entrenar: se versionan con DVC, porque suelen ser muy grandes y cambian con frecuencia.
- Los modelos y artefactos generados (pesos, parámetros, métricas, artefactos de entrenamiento): se versionan con MLflow, que además registra el historial de corridas y permite comparar experimentos.

**¿Por qué no alcanza con Git para versionar los datos?**

Git funciona bien para código, pero no para datasets grandes y pesados. Los datos de machine learning suelen ser enormes, binarios o difíciles de difundir, y hacer clones o pulls de un repositorio con esos archivos se vuelve lento e impráctico. Además, para reproducibilidad necesitamos saber exactamente qué versión del dataset usó cada entrenamiento; Git no está diseñado para manejar ese flujo de datos de forma eficiente ni para registrar artefactos de ML con metadatos y métricas asociados.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Tendría que revisar qué diferencias hay entre las dos corridas en los elementos que afectan el resultado final. En particular, me interesaría saber:

- Los hiperparámetros exactos del modelo (por ejemplo, número de árboles, profundidad, learning rate, subsample, random_state, etc.).
- La semilla usada para la división train/test y si ambos entrenaron con la misma partición.
- Si usaron exactamente el mismo conjunto de datos, y si la preparación/feature engineering fue idéntica (limpieza, codificación, imputación, escalado, columnas usadas).
- Si hubo cambios en el archivo de datos o en la versión del dataset.
- La versión del código y de las librerías que se usaron para entrenar (pandas, scikit-learn, etc.), porque eso puede cambiar el comportamiento del algoritmo.
- Si ambos guardaron y cargaron el mismo modelo o si se percibió la misma métrica pero con artefactos distintos.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Ambos tienen razón, pero sobre distintos conjuntos de prueba: el que usa `random_state=42` con la división por defecto tiene `mae_prueba = 1.2754`, y el que usa la división con `random_state=7` tiene `mae_prueba = 1.1867`. La salida muestra que la diferencia no viene del algoritmo sino de la partición de datos. Lo que sí te permite saberlo es el número del `mae_prueba` y la semilla de la división; lo que no te permite saberlo es cuál de los dos es el "verdadero" error general si no comparas sobre la misma partición. Para una comparación justa, ambos modelos tienen que evaluarse sobre la misma división.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La diferencia se explica principalmente por la partición de entrenamiento/prueba: los dos modelos son el mismo algoritmo, con los mismos hiperparámetros, pero no ven la misma muestra. Lo que cambia en la tabla es la semilla de la división (`params.semilla_division`), y también la huella del dataset (`params.huella_datos`) si cambia el archivo. En otras palabras, la clave no es la arquitectura del gradient boosting sino la forma en que se dividieron los datos. En la tabla del paso 5 esperaba ver `params.semilla_division` distinto (`42` vs `7`) y el mismo `huella_datos` en ambos, porque es el mismo CSV.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Para poder reproducir predicciones en el futuro, servir el modelo en producción, compararlo con versiones posteriores, auditarlos y volver a calcular métricas sin reentrenar. Los parámetros y las métricas describen la corrida, pero el modelo entrenado es el artefacto que permite predecir sobre nuevos datos y validar que el objeto guardado es exactamente el que se usó para esa corrida.

**Lo que mostró la tabla del paso 5:**

La tabla muestra tres corridas con el mismo dataset y la misma huella `4800204d50c0`, pero con distintas divisiones:

- `bosque aleatorio`: `mae_prueba = 1.3016`, `semilla_division = 42`
- `gradient boosting`: `mae_prueba = 1.2754`, `semilla_division = 42`
- `gradient boosting` (compañero): `mae_prueba = 1.1867`, `semilla_division = 7`

La huella del archivo es la misma porque ambos usan el mismo `viajes_features.csv`, la diferencia está en la semilla de la división, no en que el CSV haya cambiado.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

El ganador es `red neuronal`, con `mae_prueba = 1.1695`. El segundo es `gradient boosting`, con `mae_prueba = 1.2754`. La diferencia es:

- `1.275352 - 1.169505 = 0.105847`

Es decir, la red neuronal gana por aproximadamente `0.106` puntos de MAE en prueba.

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Te garantiza que, si se carga el mismo artefacto guardado en la corrida ganadora y se evalúa con los mismos datos de prueba y la misma preparación, el error reproducido debe coincidir exactamente con el número registrado, porque el modelo y el pipeline se almacenan como artefactos de la corrida. Lo que no te garantiza es que el modelo será mejor fuera de ese conjunto de prueba, o que seguirá siendo el mejor si cambian los datos o la división. Tampoco te garantiza nada sobre generalización a futuro: solo te garantiza reproducibilidad de la corrida registrada.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Porque la huella del archivo cambió: en MLflow aparecen los parámetros `params.huella_datos` y `params.semilla_division`. Si cambió una fila del CSV, la huella cambia y la corrida nueva queda identificada como una corrida distinta, aunque el nombre del modelo y la métrica puedan parecer similares. Se puede ver que la huella no coincide con la de la corrida previa y que la nueva ejecución es una corrida diferente.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

El registro de la red neuronal incluye mucha más información que una tabla escrita a mano: los hiperparámetros exactos (`hidden_layer_sizes`, `max_iter`, `random_state`), el tiempo de entrenamiento, la semilla de la división, la huella del archivo de datos, el modelo completo como artefacto, la fecha/hora de la corrida y el `run_id` asociado. Eso permite rastrear qué versión del modelo y del dataset generó cada métrica, y volver a cargarlo exactamente igual.

## Ejercicio 3: el registro de tu proyecto

Usé el proyecto de supermercado de la Semana 9, con el pipeline temporal de [`pipeline_proyecto.py`](../semana-09-sesion-02/plantilla-pipeline/pipeline_proyecto.py). La muestra reproducible contiene 20.000 filas de las primeras 100.000 del archivo `obt_sales.csv`; la división temporal cortó en 2016-04-03, con 16.838 filas de entrenamiento y 3.162 de prueba. El MD5 registrado del archivo es `e4bc0d6138eeb7913b301a3ca361275a`.

### Entregable 1: `experimentos_proyecto.ipynb`

El notebook [`experimentos_proyecto.ipynb`](../semana-09-sesion-02/plantilla-pipeline/experimentos_proyecto.ipynb) está junto al pipeline. Importa de allí la preparación y la división y registra una corrida por modelo en el experimento MLflow `supermercado-units`. Registra:

- Los hiperparámetros del modelo.
- La huella MD5 de los datos, la semilla 42 usada para muestrear, el corte temporal y los tamaños de las particiones.
- MAE en entrenamiento y prueba y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, marca la corrida elegida con `candidato=final`, carga el pipeline desde MLflow y comprueba que reproduce la métrica de prueba registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla [`corridas_proyecto.csv`](../semana-09-sesion-02/plantilla-pipeline/corridas_proyecto.csv) fue exportada desde el mismo notebook con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)`. Contiene las cuatro corridas y sus `run_id`, parámetros, métricas y etiquetas. MLflow conserva el historial: si se vuelve a ejecutar el notebook, hay que borrar las corridas de prueba del experimento en la interfaz de MLflow antes de generar la tabla final para que contenga una sola corrida por modelo.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

Elegí la **línea base (`DummyRegressor` con mediana)**: tuvo el menor MAE en la prueba temporal. La corrida quedó marcada `candidato=final` en MLflow:

| Modelo | MAE entrenamiento | MAE prueba | Entrenamiento | `run_id` |
|---|---:|---:|---:|---|
| Línea base (mediana) **final** | 1.2821 | **1.3956** | 0.12 s | `cdba8ec301e24920afa0e037d356b826` |
| Ridge (`alpha=10`) | 1.6151 | 1.6383 | 0.04 s | `79490e7e5720453d91ccfe92c2bb0e2c` |
| Gradient Boosting (`random_state=42`) | 1.6089 | 1.6506 | 0.89 s | `3cec1669069848458b4790a439b14d88` |
| Red neuronal (64 neuronas, máximo 1000 iteraciones) | 1.6553 | 1.7584 | 7.95 s | `13db461d7235475ea5f6d2ff31cbeca3` |

Aunque los modelos entrenados son más complejos, ninguno mejora la referencia en esta prueba: la línea base reduce el MAE en 0.2426 frente a Ridge, 0.2549 frente a Gradient Boosting y 0.3627 frente a la red neuronal. La corrida final guardada en MLflow reprodujo el mismo MAE al cargar el artefacto: `1.3956356736242885`.

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita el mismo `obt_sales.csv`, la misma muestra de 20.000 filas extraída de las primeras 100.000 con semilla 42, el corte temporal en 2016-04-03, el pipeline y las mismas versiones de las librerías. MLflow conserva los hiperparámetros, la huella MD5 `e4bc0d6138eeb7913b301a3ca361275a`, el corte, los tamaños de las particiones, las métricas y el pipeline entrenado. No fija automáticamente el código fuente ni todo el entorno; para reproducibilidad completa también hay que conservar la versión del código y las dependencias.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Sí cambiaría bastante. En el ejercicio 1, el mismo algoritmo `GradientBoostingRegressor` obtuvo un `mae_prueba` de `1.2754` con `semilla_division = 42` y `1.1867` con `semilla_division = 7`, una diferencia de `0.0887` en prueba. Eso se ve directamente en el registro porque cada corrida guarda `params.semilla_division` y `metrics.mae_prueba`. Es decir, la división influye en la métrica y eso queda explícito en MLflow; no es un dato que se pierda al guardar la corrida.
