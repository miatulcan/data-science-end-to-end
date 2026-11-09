# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

La paradoja aparece cuando una clase es mucho más frecuente que la otra: un clasificador puede tener una accuracy alta prediciendo siempre la clase mayoritaria y, aun así, no detectar ningún caso de la clase minoritaria. En ese caso conviene mirar precision y recall, además de F1 o métricas basadas en la curva precision-recall, y elegir según el costo de falsos positivos y falsos negativos.

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

Optimizar un indicador puede mejorar el número sin mejorar lo que realmente importa; por ejemplo, enseñar para el examen puede subir el puntaje sin aumentar en la misma medida el aprendizaje. En machine learning, elegir solo por accuracy puede premiar un modelo que ignora los fraudes. Hay que acompañar la métrica con el objetivo real del negocio, otros indicadores y una evaluación que represente el uso futuro.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

En el ejercicio de taxis de la sesión pasada, el gradient boosting obtuvo MAE **1.2754** con `random_state=42` y **1.1867** con `random_state=7`: cambió **0.0887**. MLflow registró la semilla y el MAE de cada corrida, así que se ve que la diferencia corresponde a la partición; comparar modelos requiere la misma partición.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Espero que sean relativamente parecidos, aunque no idénticos: cada pliegue usa una muestra distinta. En la corrida, el MAE promedio fue **1.2288**, con desviación estándar **0.0361**; los cinco errores redondeados fueron **1.28, 1.24, 1.16, 1.24 y 1.23**. El rango observado fue 0.12.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Esperaría que el error temporal fuera mayor, porque se valida en viajes posteriores y no se mezclan observaciones futuras en el entrenamiento. El futuro puede tener condiciones distintas. En esta corrida dio **1.3099 ± 0.0693**, frente a **1.2288 ± 0.0361** al azar; aquí el promedio temporal fue 0.0811 mayor, aunque no todos los pliegues temporales fueron peores que todos los aleatorios.

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El peor fue el pliegue 5, que dejó fuera viajes con origen en **Brooklyn** y **JFK Airport**. Su MAE fue **6.43**, frente a un promedio de **2.2996** entre pliegues y desviación **2.0648**. Al excluir zonas completas, el modelo debe generalizar a orígenes que no vio durante el entrenamiento.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El porcentaje de viajes subestimados en más de 20% responde directamente cuántos casos cumplen la condición del reclamo. En el conjunto de prueba fue **4.37%** (26 de 595). El MAE resume el error absoluto típico y el RMSE penaliza más los errores grandes, pero ninguno cuenta por sí solo los reclamos según ese umbral.

**Lo que confirmó la tabla, esquema por esquema:**

| Esquema/modelo | MAE promedio | Desviación estándar | Peor pliegue (MAE) |
|---|---:|---:|---:|
| Aleatorio, Gradient Boosting | 1.2288 | 0.0361 | 1.28 |
| Aleatorio, bosque aleatorio | 1.2880 | 0.0291 | 1.34 |
| Temporal, Gradient Boosting | 1.3099 | 0.0693 | 1.40 |
| Por zona, Gradient Boosting | 2.2996 | 2.0648 | 6.43 |

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

La línea base tuvo accuracy **0.9508** y recall **0.0000**: acierta los **5.705** pagos legítimos, pero no detecta ninguno de los **295** fraudes.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

El gradient boosting alcanza accuracy **0.9712** (una mejora de **2.04 puntos porcentuales** sobre 0.9508) y recall **0.4610** (una mejora de **46.10 puntos** sobre cero). Para la empresa, el recall y el costo de errores describen mejor lo que gana: la accuracy alta de la línea base ocultaba que no detectaba fraudes.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con umbral **0.5**, dejó pasar **159 fraudes** y marcó **14 pagos legítimos** como fraude. El paso 5 eligió umbral **0.1**: dejó pasar **122 fraudes** y generó **222 falsas alarmas**. Con los costos indicados, el costo total de ese umbral fue **USD 5,546** en los 6.000 pagos.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

El umbral sube: si revisar una falsa alarma cuesta más, conviene reducir las alertas. Al cambiar el costo de falsa alarma de $3 a $20, el umbral elegido entre los candidatos subió de **0.10 a 0.50**. En 0.50 quedaron **159 fraudes sin detectar** y **14 falsas alarmas**. Esto minimiza el costo definido, no el número de fraudes por sí solo.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

El notebook [`validacion_proyecto.ipynb`](../semana-09-sesion-02/plantilla-pipeline/validacion_proyecto.ipynb) compara la línea base final de Semana 11.1 y Ridge, el segundo mejor según su MAE en la prueba. Exporta los resultados por pliegue y los resúmenes a [`resultados_validacion_proyecto.csv`](../semana-09-sesion-02/plantilla-pipeline/resultados_validacion_proyecto.csv).

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

Para el ejemplo se supone que subestimar cuesta **$2 por unidad** en margen de venta perdido y sobreestimar cuesta **$0.50 por unidad** en inventario/merma. Son supuestos ilustrativos, no datos contables observados. El notebook calcula el costo asimétrico medio por fila junto con el MAE.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

La validación **temporal** corresponde: el objetivo es pronosticar unidades de fechas futuras, no completar fechas mezcladas al azar. Las mismas combinaciones tienda-categoría aparecen repetidas en distintos días, así que se evalúa con fechas posteriores a las de entrenamiento. El notebook divide por fechas completas para evitar que observaciones del mismo día queden en ambos lados del pliegue.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

Para MAE, no cambió el ganador: la línea base siguió siendo mejor que Ridge tanto al azar (**1.3000 ± 0.0508** frente a **1.6205 ± 0.0534**) como temporalmente (**1.2766 ± 0.2238** frente a **3.5118 ± 4.4952**). Sin embargo, la métrica de costo sí invirtió el orden en la validación aleatoria: Ridge costó **$2.0258 por fila** frente a **$2.6001** para la línea base; en la temporal, la línea base costó **$2.5531** frente a **$6.2228** para Ridge. Para predecir el futuro, tomo la validación temporal como referencia principal.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

En aleatorio, la diferencia de MAE medio fue **0.3204**, mayor que las desviaciones de ambos modelos (aprox. **0.05**). En temporal, la diferencia fue **2.2352**, menor que la desviación de Ridge (**4.4952**), causada por su MAE de **11.5474** en el pliegue 2. Aun así, la línea base ganó en cada uno de los cinco pliegues temporales; la mantengo como candidata, con cautela porque la muestra solo cubre 20 fechas y Ridge tuvo un pliegue atípico.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

Elegí un costo asimétrico medio por fila: **$2 por unidad subestimada** y **$0.50 por unidad sobreestimada**. A diferencia del MAE, pondera más quedarse corto y expresa una pérdida aproximada en dólares por predicción. Con estos supuestos, en temporal la línea base obtiene **$2.5531 por fila**, frente a **$6.2228** de Ridge. Los costos son ilustrativos y deben sustituirse por margen y costos reales antes de tomar decisiones de inventario.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La cola de fraude tiene que evaluarse en línea, antes de autorizar o liquidar el pago: se necesita una respuesta en segundos o menos para decidir si aprobarlo o retenerlo. Calcularla una vez por noche dejaría pasar transacciones fraudulentas durante el día. Las tarifas de viajes ya programados sí se pueden estimar por lotes durante la noche para tenerlas listas por la mañana; no se pueden precalcular todos los viajes posibles que aún no se solicitaron, así que una solicitud inmediata necesita una predicción en línea.
