# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

El aprendizaje automático clásico conviene para datos estructurados y tareas específicas, como predecir una tarifa a partir de la distancia, la hora y las zonas del viaje. Un LLM es más adecuado para comprender y generar lenguaje, y para trabajar con instrucciones o entradas de texto flexible.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

Para una app de taxis, pesa más la explicabilidad: sería útil poder justificar una tarifa al conductor o al pasajero. También importan el costo por predicción y que el modelo funcione bien con los datos estructurados de los viajes.

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Mediría el tiempo de entrenamiento y de respuesta, el costo de operación (incluidas las llamadas y los tokens si se usa una API), la consistencia de las predicciones y qué tan fácil es explicar cada tarifa. Compararía las tres opciones con los mismos viajes de prueba.

## Ejercicio 1: una red neuronal frente al ensamblaje

Archivo: [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

`ConvergenceWarning` indica que el entrenamiento alcanzó el límite de iteraciones antes de converger, la red quizá aún podía mejorar. Al permitir hasta 1000 iteraciones, esperaría darle más oportunidad de aprender y reducir el error, aunque tardaría más. En la corrida con ese límite no apareció la advertencia.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

Aprende 1729 parámetros: 25 × 64 pesos de entrada a la capa oculta, 64 interceptos de esa capa, 64 pesos hacia la salida y 1 intercepto de salida.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

Esperaría que la columna más importante fuera `distancia_km`, porque probablemente influye mucho en la tarifa. En esta corrida fue la más importante. El gradient boosting permite analizar las predicciones con herramientas como SHAP, sus importancias por sí solas muestran tendencias globales, no la explicación de un viaje individual.

**Lo que confirmó la salida, TODO por TODO:**

En esta corrida, el gradient boosting obtuvo un MAE de **1.0705** en entrenamiento y **1.2754** en prueba, y tardó **0.295 s**. La red neuronal obtuvo un MAE de **1.0882** en entrenamiento y **1.1695** en prueba, y tardó **3.983 s**. La red aprendió **1729 parámetros**. Las tres columnas más importantes para el ensamblaje fueron `distancia_km` (**0.9824**), `es_tarifa_fija` (**0.0120**) y `es_hora_pico` (**0.0028**). Los tiempos pueden variar entre ejecuciones.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Archivo: [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

LLM: el asistente de esta conversación; la interfaz no expone el nombre ni la versión exacta del modelo. Las 20 estimaciones fueron generadas aquí, no mediante una llamada a una API externa. Para comparar costos de API, uso solo como referencia GPT-4o mini a $0.15 por millón de tokens de entrada.

| | Error en los 20 viajes (MAE, USD) | Costo/tiempo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.8957 | 0.27 s en esta corrida local; sin costo directo de API | Sin costo de API local; infraestructura no contabilizada | Se pueden explicar contribuciones con herramientas como SHAP; `feature_importances_` por sí sola es global |
| Red neuronal | 0.9637 | 4.45 s en esta corrida local; sin costo directo de API | Sin costo de API local; infraestructura no contabilizada | No fácilmente; es menos interpretable |
| LLM | 0.9235 | No se entrenó; es un modelo preentrenado | $0.0544 estimados con la referencia de precio de API indicada | Puede redactar una explicación, pero no garantiza que refleje fielmente cómo obtuvo la cifra |

El costo del LLM es una estimación basada en el tamaño del prompt, un token cada cuatro caracteres y 500 prompts al mes. No es un cargo real de esta conversación.

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

No: al repetir el prompt en una conversación nueva, cambiaron 17 de los 20 números. Por ejemplo, para el primer viaje la estimación pasó de $14.50 a $14.33, y para el tercero de $8.50 a $10.20. El MAE pasó de $0.9235 en la primera corrida a $0.9945 en la segunda, sobre los mismos 20 viajes. Las dos listas están guardadas en [`starter/respuestas_llm.txt`](./starter/respuestas_llm.txt) y [`starter/respuestas_llm_segunda_corrida.txt`](./starter/respuestas_llm_segunda_corrida.txt). Esta variación podría hacer que la app cobre tarifas distintas por el mismo viaje, dificultando la reproducibilidad, las auditorías y la explicación de cobros a pasajeros.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor fue un viaje de JFK Airport a JFK Airport de 1.01 km: la estimación fue $8.00 frente a una tarifa real de $11.46, un error absoluto de $3.46. Es una ruta inusual por ser muy corta y tener origen y destino en el aeropuerto. En las 2380 filas de entrenamiento había 7 viajes JFK–JFK; el gradient boosting predijo $12.74 para este caso, cerca de la tarifa real. En este conjunto sí había ejemplos que le permitieron aprender un valor razonable para esa combinación.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

No es imposible medir el LLM sobre los 595: se podría hacer por lotes, pero habría que enviar muchas más filas y repetir los ejemplos de contexto, lo que aumenta el tiempo, el costo de tokens y la posibilidad de errores de formato en las respuestas. Para este ejercicio se usa una muestra manejable de 20 viajes y se compara con los modelos clásicos sobre esas mismas filas; por eso su MAE (0.9235 en la primera corrida) no se debe comparar directamente con el MAE calculado sobre los 595.

Una comparación sobre 20 es orientativa, no muy concluyente: cada viaje pesa 1/20 del MAE y unos pocos casos atípicos pueden cambiar bastante el resultado. Además, la muestra aleatoria puede no representar todos los tipos de viaje y el LLM puede variar entre conversaciones. Para decidir qué modelo conviene, harían falta más filas de prueba, una evaluación repetible y la misma métrica calculada sobre exactamente los mismos casos.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.py`

Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

Usé el proyecto de tarifas de taxi y el archivo [`starter/viajes_features.csv`](./starter/viajes_features.csv). Conservé la preparación de la Semana 9 y la división aleatoria 80/20 (`random_state=42`): 2380 viajes para entrenamiento y 595 para prueba. Para la comparación del LLM, los tres resultados corresponden a los mismos 10 viajes de prueba (`random_state=7`); el prompt incluye 30 ejemplos de entrenamiento.

| Opción | MAE en los 595 de prueba (USD) | MAE en los mismos 10 viajes (USD) | Costo/tiempo de entrenamiento | Costo mensual estimado para el LLM | Explicabilidad |
|---|---:|---:|---|---:|---|
| Gradient boosting | 1.2754 | 0.7794 | 0.24 s local | No aplica si se ejecuta localmente; infraestructura no contabilizada | Se puede explicar con contribuciones de features (p. ej., SHAP) |
| Red neuronal (64 neuronas) | 1.1695 | 0.7251 | 4.08 s local; 1729 parámetros | No aplica si se ejecuta localmente; infraestructura no contabilizada | Baja; no ofrece una explicación sencilla por predicción |
| LLM | No medido en los 595 | 0.4960 | No se entrenó aquí; modelo preentrenado | $0.0913 estimados con la referencia de GPT-4o mini API | Puede redactar una razón, pero no es garantía de una explicación fiel |

Las estimaciones del LLM para esos 10 viajes se generaron en esta conversación y no mediante una llamada externa a GPT-4o mini; por tanto, el costo mensual es solo una referencia hipotética de API, y el MAE sobre 10 casos no es comparable con el MAE sobre los 595.

Usaría gradient boosting para una primera versión de la app: su MAE en la prueba completa fue $1.2754, unos $0.106 peor que la red neuronal ($1.1695), pero entrenó en aproximadamente 0.24 s frente a 4.08 s y permite analizar contribuciones por viaje con herramientas de explicación. No elegiría el LLM basándome en su MAE menor de 10 casos: esa muestra es pequeña y la cifra no se midió en toda la prueba. Antes de cobrar tarifas reales, validaría el modelo con más datos y una explicación consistente para cada cobro.
## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Para encontrar la causa, compararía el registro reproducible de cada corrida:

- **Datos y features:** versión o hash del archivo, filas incluidas, columnas usadas, tratamiento de faltantes y transformaciones aplicadas.
- **División de datos:** método, proporción, `random_state` y, si aplica, orden temporal. Hay que confirmar que ambos evaluaron exactamente las mismas filas de prueba.
- **Modelo y entrenamiento:** clase exacta (`GradientBoostingRegressor`), hiperparámetros, semilla aleatoria y cualquier búsqueda o ajuste realizado.
- **Evaluación:** métrica, conjunto sobre el que se calculó y opciones de esa métrica (por ejemplo, MAE en dólares y no otra escala).
- **Código y entorno:** commit o versión del script, versiones de Python, scikit-learn y dependencias, además de plataforma si pudiera afectar la ejecución.

Primero comprobaría que los dos usaron las mismas filas de entrenamiento y prueba y la misma métrica. Después compararía el pipeline, los hiperparámetros y las versiones de código y librerías. Si alguna de esas piezas difiere, no es exactamente la misma corrida aunque el nombre del modelo y el archivo de datos coincidan.
