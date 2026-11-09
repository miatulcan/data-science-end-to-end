# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el pipeline de tu proyecto de la Semana 9, Sesión 2.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje combina las predicciones de varios modelos para producir una predicción final. En bagging, los modelos se entrenan en paralelo con distintas muestras de los datos y luego se combinan (por ejemplo, promediando sus predicciones). En boosting, se entrenan secuencialmente: cada modelo nuevo busca corregir los errores de los anteriores y sus predicciones se combinan, normalmente con ponderaciones.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Cada árbol puede cometer errores distintos. Al promediar, los errores particulares tienden a compensarse, lo que reduce la variabilidad de la predicción, esto funciona mejor cuando los árboles no cometen exactamente los mismos errores.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Ese modelo probablemente se está sobreajustando: se adapta demasiado a los ejemplos de entrenamiento, incluso a sus particularidades o ruido, y generaliza peor a datos no vistos. La diferencia entre ambos errores es evidencia de esa brecha de generalización.

## Ejercicio 1: línea base, un árbol y un bosque

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. El ejercicio compara una línea base, un árbol sin límite, el mismo árbol limitado y un bosque aleatorio:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

En esta ejecución, la línea base tuvo un MAE de **13.1998** en entrenamiento y **12.5308** en prueba, es decir, sus predicciones se alejaron de la tarifa real en unas 13 unidades monetarias, en promedio. Aunque no usa columnas, sirve como referencia sencilla: permite comprobar si los modelos que sí usan las características realmente aportan valor y detectar problemas en la evaluación.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

El árbol sin límite obtuvo MAE **0.0029** en entrenamiento y **1.7017** en prueba. El error casi nulo en entrenamiento confirma que memoriza los ejemplos; la diferencia en prueba muestra que no generaliza igual de bien.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Con `max_depth=8`, el árbol obtuvo MAE **1.0098** en entrenamiento y **1.2728** en prueba: el error de entrenamiento subió, pero el de prueba bajó y la brecha se redujo a **0.2630**. El bosque de 200 árboles obtuvo **0.4739** en entrenamiento y **1.3016** en prueba, con brecha **0.8276**. En esta corrida, el árbol limitado tuvo el menor error de prueba; el bosque mejoró frente al árbol sin límite, pero no superó al árbol limitado.

**Lo que confirmó la tabla, TODO por TODO:**

Con los cuatro modelos implementados, la tabla dio:

| Modelo (TODO) | MAE entrenamiento | MAE prueba | Brecha |
|---|---:|---:|---:|
| Línea base (TODO 1) | 13.1998 | 12.5308 | -0.6690 |
| Árbol sin límite (TODO 2) | 0.0029 | 1.7017 | 1.6988 |
| Árbol `max_depth=8` (TODO 3) | 1.0098 | 1.2728 | 0.2630 |
| Bosque aleatorio (TODO 4) | 0.4739 | 1.3016 | 0.8276 |

El árbol sin límite sobreajusta: casi memoriza el entrenamiento, pero tiene el mayor MAE en prueba. Limitar la profundidad reduce esa brecha y produjo el menor MAE de prueba de los cuatro modelos. El bosque también reduce el error de prueba frente al árbol sin límite, aunque en esta partición queda ligeramente por detrás del árbol limitado. Los tiempos pueden variar entre ejecuciones.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D)
**Tu razonamiento:**
**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Archivo: [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_regularizacion.py
```

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Recibe 3275 columnas y hay 2380 filas de entrenamiento. Eso significa que el modelo tiene muchos más parámetros que ejemplos de entrenamiento. Al crear todas las interacciones hasta grado 3, el modelo se vuelve demasiado complejo, memoriza el entrenamiento y falla en prueba. Por eso el MAE en prueba se dispara.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

- `lineal grado 3`: entrenamiento 0.8922, prueba 1695287900.1597
- `ridge grado 3`: entrenamiento 1.0272, prueba 1.3209

Ridge empeora un poco en entrenamiento porque penaliza los coeficientes grandes y deja de ajustar los datos tan exactamente. Pero mejora mucho en prueba porque reduce el sobreajuste y generaliza mejor.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

`gradient boosting` fue el mejor en prueba, con MAE de 1.2754. La línea base tuvo MAE de 12.5308. La mejora fue de 11.2554 puntos, aproximadamente un 89.9% mejor que la línea base.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

Porque la línea base siempre predice la mediana, y por casualidad el conjunto de prueba puede ser más fácil para esa constante que el conjunto de entrenamiento. La brecha negativa no significa que el modelo haya aprendido; solo significa que la partición aleatoria hizo que la prueba fuera más favorable.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el script tú.

### Entregable 1: `modelos_proyecto.py`

Un script, en la carpeta de tu proyecto, que entrene con tu pipeline:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio.
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

La línea base es `DummyRegressor(strategy="median")` porque `UNITS` es una variable numérica y estamos resolviendo una regresión. La mediana es una referencia robusta y simple que no usa las características; así comprobamos si los modelos más complejos realmente agregan valor.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

No aparece evidencia de sobreajuste fuerte: las diferencias entre MAE de entrenamiento y prueba son pequeñas. En la tabla del proyecto:

- `linea base`: entrenamiento 1.2821, prueba 1.3956, brecha 0.1135
- `ridge alpha 10`: entrenamiento 1.6151, prueba 1.6383, brecha 0.0231
- `gradient boosting`: entrenamiento 1.6089, prueba 1.6506, brecha 0.0417

En los tres casos el error de prueba es un poco mayor que el de entrenamiento, pero no hay una brecha grande. Además, la línea base obtiene el menor MAE en prueba.

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

Apliqué `Ridge(alpha=10.0)`, que penaliza coeficientes grandes. Su brecha fue pequeña (0.0231), pero su MAE en prueba (1.6383) fue mayor que el de la línea base (1.3956). Por tanto, en esta muestra la regularización no hizo que Ridge superara la referencia simple.

**¿Cuánto mejora tu mejor modelo a la línea base?**

Ningún modelo entrenado mejoró la línea base en prueba. La línea base obtuvo MAE 1.3956; Ridge obtuvo 1.6383 y gradient boosting 1.6506. El mejor resultado general de esta comparación es, por tanto, la línea base. Gradient boosting tuvo un error 0.2549 mayor que ella. Los archivos del proyecto de supermercado están en [`proyecto-supermercado/`](./proyecto-supermercado/): allí están `modelos_proyecto.py` y el CSV producido.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error, hay que medir el costo de operación, el tiempo de entrenamiento y predicción, la consistencia de las respuestas, la explicabilidad y la latencia. En una app de taxis también importa que el modelo sea rápido, económico y que pueda explicar por qué una tarifa fue calculada.
