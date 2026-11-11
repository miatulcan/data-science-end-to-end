# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

En batch, el modelo predice para un conjunto de casos y deja los resultados disponibles para consultarlos después, como un pronóstico nocturno de las ventas del día siguiente. En línea, calcula una predicción cuando llega una solicitud, como puntuar un pago durante su autorización.

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

El resultado batch puede estar desactualizado y no sirve para casos nuevos hasta la siguiente corrida. El servicio online responde al momento, pero requiere una API disponible, baja latencia, monitoreo y más infraestructura y operación.

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

Las tarifas de viajes ya programados se pueden calcular por la noche, aunque no se pueden precalcular todos los viajes posibles porque origen y destino pueden variar. El modelo de fraude debe responder durante la autorización del pago, idealmente en menos de 100 milisegundos, para decidir si se aprueba o se retiene.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Las otras dos pruebas todavía no tenían `assert`, así que pytest las marcaba como aprobadas sin comprobar las predicciones. La prueba de columnas sí detectaba que faltaban `tarifa_estimada` y `version_modelo`.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Debe estimar una tarifa para cada viaje y conservar una fila por caso. El pipeline imputa los valores numéricos faltantes con la mediana y las categorías faltantes con la categoría más frecuente aprendida durante el entrenamiento.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Permite rastrear qué versión produjo cada predicción, incluso si luego las filas se separan o combinan con otros resultados. Así se pueden auditar decisiones y reproducir resultados.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

La prueba del TODO 4 falla si se omite o reordena un viaje, o si se devuelve una cantidad de filas distinta. La del TODO 5 falla si el viaje sin pasajeros no recibe tarifa o si alguna fila no conserva la versión del modelo.

**Lo que mostró el lote:**

En esta ejecución, la versión fue `8cf725276958`. Las tarifas estuvieron entre 4.67 y 70.26. Los cuatro viajes sin pasajeros recibieron estimaciones de 33.94, 29.88, 25.80 y 17.58.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

Quedaron 26 pagos en la cola con el umbral 0.10. Con la proporción histórica de 295 fraudes entre 6000 pagos, esperaría aproximadamente 14.75, es decir, unos 15 fraudes en 300 pagos.

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

La empresa se entera tarde y no puede detener la autorización de un pago sospechoso en el momento. Esto importa si el dinero se transfiere o retira antes de que el equipo revise la cola. La evaluación en línea sería útil si la decisión de aprobar, retener o rechazar debe tomarse durante la transacción.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

La prueba es pequeña, rápida y determinista. Aísla la regla del umbral y el orden de la cola, por lo que detecta errores en esa lógica sin depender del entrenamiento, de los datos reales ni de las variaciones del modelo.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Con el umbral 0.50 quedan 13 pagos. Las pruebas siguen pasando porque la prueba unitaria de la cola usa una tabla construida a mano y le pasa explícitamente el umbral 0.10. No verifica el valor de la constante global ni el tamaño de la cola del archivo real.

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

https://github.com/miatulcan/data-science-end-to-end/actions/runs/43048070888


**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

Elegí batch para pronosticar las ventas por combinación de fecha, tienda y categoría. El equipo de inventario puede esperar hasta una noche, con una latencia objetivo inferior a 24 horas.

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

Cambié temporalmente el valor esperado de `es_fin_de_semana` de `[1, 0]` a `[1, 1]`. GitHub Actions marcó en rojo la ejecución porque falló `test_features_de_calendario_y_tienda_categoria`: la función devolvió `[1, 0]`, como correspondía a las fechas de prueba. Restauré el valor esperado a `[1, 0]`, subí el arreglo y la siguiente ejecución terminó en verde. Vi ambos resultados en la pestaña **Actions**, al abrir las ejecuciones de los commits.

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Haría falta un servicio de predicción en línea, normalmente una API con un endpoint disponible para la app. La app enviaría los datos del viaje, como origen, destino, distancia, hora, cantidad de pasajeros y método de pago, y recibiría la tarifa estimada en la respuesta.
