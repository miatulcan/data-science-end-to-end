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

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

En esta ejecución, la línea base tuvo un MAE de **13.5877** en entrenamiento y **12.9215** en prueba, es decir, sus predicciones se alejaron de la tarifa real en unas 13 unidades monetarias, en promedio. Aunque no usa columnas, sirve como referencia sencilla: permite comprobar si los modelos que sí usan las características realmente aportan valor y detectar problemas en la evaluación.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

Espero un error muy bajo —posiblemente cercano a cero— en entrenamiento, porque un árbol sin límite puede memorizar los ejemplos. En prueba espero un error mayor: las divisiones tan específicas pueden ajustarse al ruido y no generalizar a viajes que no vio.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad, espero que el árbol deje de memorizar tanto: puede subir el error de entrenamiento, pero bajar el de prueba y reducirse la brecha entre ambos. Con 200 árboles, el promedio suele reducir la variabilidad de un árbol individual y mejorar la generalización; espero un error de prueba competitivo y una brecha menor que la del árbol sin límite.

**Lo que confirmó la tabla, TODO por TODO:**

En la ejecución que hice, antes de completar los TODO, la tabla dio lo mismo para los cuatro renglones:

| Modelo (TODO) | MAE entrenamiento | MAE prueba | Brecha |
|---|---:|---:|---:|
| Línea base (TODO 1) | 13.5877 | 12.9215 | -0.6662 |
| Árbol sin límite (TODO 2) | 13.5877 | 12.9215 | -0.6662 |
| Árbol `max_depth=8` (TODO 3) | 13.5877 | 12.9215 | -0.6662 |
| Bosque aleatorio (TODO 4) | 13.5877 | 12.9215 | -0.6662 |

Esto no confirma todavía las expectativas sobre los árboles ni el bosque: el script aún tiene `DummyRegressor()` en los TODO 2, 3 y 4, por lo que los cuatro renglones están evaluando el mismo tipo de modelo. Para comparar los modelos de verdad, hay que reemplazar esos tres marcadores con las implementaciones indicadas y volver a ejecutar el script.

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

La línea base es `DummyRegressor(strategy="median")` porque el problema es regresión y la mediana es una referencia simple y válida. No usa ninguna columna, así que sirve para saber si un modelo más complejo aporta valor real.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

Sí. La evidencia está en la comparación del error en entrenamiento y prueba:

- `linea base`: entrenamiento 13.1998, prueba 12.5308, brecha -0.6690
- `ridge alpha 10`: entrenamiento 2.4939, prueba 2.5565, brecha 0.0626
- `gradient boosting`: entrenamiento 1.0705, prueba 1.2754, brecha 0.2048

La comparación del Ejercicio 2 muestra el sobreajuste extremo de la regresión con interacciones: entrenamiento 0.8922 y prueba 1695287900.1597. Eso confirma que el exceso de complejidad puede romper la generalización.

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

Apliqué `Ridge(alpha=10.0)`. La regularización penaliza coeficientes muy grandes y evita que el modelo se ajuste demasiado a ruido o relaciones muy específicas. En la tabla, esto redujo muchísimo el error de prueba respecto a la regresión no regularizada y dejó una brecha pequeña.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El mejor modelo fue `gradient boosting`, con MAE en prueba de 1.2754. La línea base estaba en 12.5308. La mejora fue 11.2554 puntos, aproximadamente un 89.9% mejor que la línea base.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error, hay que medir el costo de operación, el tiempo de entrenamiento y predicción, la consistencia de las respuestas, la explicabilidad y la latencia. En una app de taxis también importa que el modelo sea rápido, económico y que pueda explicar por qué una tarifa fue calculada.
