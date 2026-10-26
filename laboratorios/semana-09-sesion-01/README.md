# Semana 9, Sesión 1: ingeniería de características aplicada a mi proyecto

> Objetivo: crear columnas útiles para predecir las unidades vendidas (`UNITS`) por producto, tienda y día, sin usar información del futuro.

Esta resolución adapta los ejercicios de taxis al proyecto de ventas. No utiliza `viajes_taxi.csv`. La fuente es `data/processed/obt_sales.csv`, la misma de Semana 8. Los nombres de las carpetas se mantienen para conservar la estructura del laboratorio.

## Datos y muestra utilizada

El archivo pesa aproximadamente 13,48 GB. No se procesa completo: se leen únicamente las primeras 100.000 filas con `nrows=100_000` y se seleccionan 20.000 con `sample(n=20_000, random_state=42)`. Las dos plantillas usan exactamente la misma selección. El CSV original no se modifica y no se guarda otra copia grande.

La muestra contiene 10 tiendas, 30 combinaciones de tienda y categoría, y fechas entre 2012-09-27 y 2016-04-06. Ese rango no implica que estén presentes todos los días intermedios. La selección es aleatoria dentro del primer bloque, no dentro de todo el archivo: puede favorecer los períodos o grupos que aparecen primero.

Requisitos: Python, pandas y numpy. Desde la raíz del repositorio, en PowerShell:

```powershell
.venv/Scripts/python.exe laboratorios/semana-09-sesion-01/plantilla-features/features_proyecto.py
.venv/Scripts/python.exe laboratorios/semana-09-sesion-01/plantilla-ejercicio2/features_de_zona.py
```

Con otro entorno activado, sustituir `.venv/Scripts/python.exe` por `python`. Los scripts también funcionan desde otra carpeta porque resuelven el CSV a partir de su propia ubicación.

## Antes de empezar

**¿Qué diferencia a una característica que tiene sentido de negocio de una que solo mejora una métrica?**

Una columna útil tiene una explicación relacionada con el problema real y se puede calcular cuando se necesita la predicción. Una mejora numérica no basta: por ejemplo, usar los ingresos del mismo día podría ayudar muchísimo a adivinar las unidades, pero esos ingresos todavía no existen cuando hacemos el pronóstico.

**Ejemplo de conocimiento del negocio convertido en una columna.**

En este proyecto, una misma categoría puede vender distinto según la tienda. Lo represento con `tienda_categoria`, uniendo `STORE_ID` y `CAT_ID`: por ejemplo, `CA_3__FOODS`. Así el modelo puede distinguir la venta de alimentos en cada tienda.

## Pregunta de apertura

**¿Cómo entrego al modelo un patrón que no existe como columna?**

Lo convierto en una columna nueva usando datos disponibles antes de vender. En la Semana 8 se encontró diferencias entre tiendas y un contexto particular de precios ausentes. Por eso construyo `tienda_categoria` y exploro `precio_no_disponible`. No uso `UNITS` para crearlas ni decido sus valores mirando `REVENUE`.

## Ejercicio 1: fecha y calendario, adaptado a ventas

<<<<<<< HEAD
El archivo tiene fechas diarias, no horas de compra. Por eso no se pueden calcular horas pico reales. Uso `DATE` para construir el día de la semana, fin de semana y una representación cíclica del día.

**Respuesta 1: distancia numérica y distancia en un ciclo.**
=======
Archivo: [`starter/ejercicio1_features_tiempo.py`](./starter/ejercicio1_features_tiempo.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. El script ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-01/starter
python3 ejercicio1_features_tiempo.py
```

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?
>>>>>>> f9c474b3a2c2e5c9177d6177a998d779a6788b94

Entre 23 y 0 hay 23 unidades numéricas, pero en el reloj solo hay una hora. En nuestro calendario ocurre algo parecido: domingo (`6`) y lunes (`0`) están a un día, no a seis. Las columnas seno y coseno representan ese ciclo sin cortar artificialmente la semana.

<<<<<<< HEAD
**Respuesta 2: información disponible antes de predecir.**
=======
**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?
>>>>>>> f9c474b3a2c2e5c9177d6177a998d779a6788b94

En taxis, la hora pico puede aportar contexto sobre tráfico esperado sin usar la duración real. En ventas, el calendario permite identificar fines de semana antes de conocer las unidades vendidas. Esto no asegura una subida de ventas, se debe comprobar con datos.

<<<<<<< HEAD
Código aplicado, incluido en [features_proyecto.py](./plantilla-features/features_proyecto.py):

```python
fecha = pd.to_datetime(datos['DATE'])
datos['dia_semana'] = fecha.dt.dayofweek
datos['es_fin_de_semana'] = datos['dia_semana'].ge(5).astype(int)
datos['dia_sin'] = np.sin(2 * np.pi * datos['dia_semana'] / 7)
datos['dia_cos'] = np.cos(2 * np.pi * datos['dia_semana'] / 7)
```

| Grupo en la muestra | Registros | Unidades promedio |
|---|---:|---:|
| Lunes a viernes | 14.522 | 1,2383 |
| Sábado y domingo | 5.478 | 1,4637 |

En esta muestra el promedio es mayor en fin de semana. Es una relación descriptiva, no una prueba de causa ni de mejora predictiva.
=======
**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

_(tu respuesta)_

**Lo que confirmó la salida, TODO por TODO:**

_(anota la tarifa por kilómetro en cada grupo y las dos distancias del Paso 5)_
>>>>>>> f9c474b3a2c2e5c9177d6177a998d779a6788b94

## Durante el ConcepTest

**Voto:**

**Razonamiento aplicado:** 

**Cambio de voto tras discutir:** 

## Ejercicio 2: tienda, codificación y disponibilidad

Resuelto en [features_de_zona.py](./plantilla-ejercicio2/features_de_zona.py). Se conservan los cuatro pasos originales, adaptados a ventas:

1. Sustituir el indicador de tarifa fija por `precio_no_disponible` y contar sus casos y el mínimo y máximo de `UNITS`.
2. Codificar `STORE_ID` con one-hot, prefijo `tienda` y `dtype=int`.
3. Crear `tienda_ordinal` asignando números a las tiendas en orden alfabético.
4. Imprimir la disponibilidad de las 40 columnas originales al momento de predecir.

**Respuesta 1: indicador de contexto.**

<<<<<<< HEAD
Hay 1.449 registros con `precio_no_disponible = 1`. Sus unidades mínimas y máximas son 0. Cuando hay precio registrado, el promedio es 1,4016 unidades; cuando falta, es 0. El indicador permite distinguir ambos contextos. En una regresión lineal puede representar una diferencia de nivel, pero no garantiza que todas las predicciones del grupo sean exactamente cero, porque también intervienen otras columnas. La relación no demuestra que la ausencia de precio cause ventas cero.
=======
Solo o en pareja. Archivo: [`starter/ejercicio2_features_zona.py`](./starter/ejercicio2_features_zona.py), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_features_zona.py
```
>>>>>>> f9c474b3a2c2e5c9177d6177a998d779a6788b94

**Respuesta 2: one-hot frente a ordinal.**

One-hot crea 10 columnas, una por tienda. La codificación ordinal crea una sola columna, con valores de 0 a 9. Para regresión lineal usaría one-hot: los códigos ordinales harían parecer que una tienda está por encima de otra y que la distancia entre sus códigos tiene significado. Las tiendas no tienen ese orden. Si se usa intercepto, se puede dejar una categoría de referencia para evitar redundancia. Para entrenar, ajustaría un `OneHotEncoder(handle_unknown='ignore')` solamente con entrenamiento, dentro del pipeline; `get_dummies` se usa aquí para entender la representación.

**Respuesta 3: disponibilidad de las columnas.**

El momento elegido es antes de comenzar el día que se pronostica.

| Información | ¿Se puede usar? |
|---|---|
| Producto, tienda, departamento, categoría y estado | Sí, si ya están definidos en el catálogo. |
| Fecha y columnas de calendario | Sí; no dependen de las ventas del día. |
| Eventos, SNAP y cierres de calendario | Solo si estaban publicados o planificados antes del día objetivo. |
| `SELL_PRICE`, `IS_PRE_LAUNCH`, `IS_STORE_CLOSED` | Condicional: hay que verificar que procedan de un plan conocido de antemano y no de una reconstrucción posterior. |
| `UNITS` | Es el objetivo; no se usa como entrada del mismo día. |
| `REVENUE` | No: depende de las unidades que intentamos predecir. |
| `IS_SALES_SPIKE` e indicadores de posible falta de stock | Excluir por precaución: describen ventas observadas; no se ha acreditado su disponibilidad previa. |
| `_LOADED_AT` | Excluir: es un metadato de carga. |

La decisión individual de las 40 columnas está en [resultados_codificacion.json](./plantilla-ejercicio2/resultados_codificacion.json). Usar datos conocidos después produciría fuga de información: una evaluación demasiado optimista que no representa el uso real.

## Ejercicio 3: dos características para mi proyecto

### Paso 1: qué predigo y cuándo

Predigo `UNITS` por producto, tienda y día antes de que comience ese día. Conozco el catálogo, la tienda y la fecha. El precio solo se considera conocido si es un precio planificado disponible en ese instante. No conozco todavía las ventas, los ingresos ni los diagnósticos realizados con las ventas de ese día.

### Paso 2: construcción

La solución ejecutable está en [features_proyecto.py](./plantilla-features/features_proyecto.py).

```python
datos['precio_no_disponible'] = datos['SELL_PRICE'].isna().astype(int)
datos['tienda_categoria'] = datos['STORE_ID'] + '__' + datos['CAT_ID']
```

Las columnas del ejercicio de calendario también se calculan allí, como práctica adicional. No se entrena todavía un modelo.

### Paso 3: justificación y evidencia

**Característica 1: `precio_no_disponible`.**

Surge del hallazgo 2 de Semana 8: los precios ausentes estaban relacionados con registros de pre-lanzamiento. No conviene reemplazar automáticamente ese precio por cero como si fuera un precio real. El indicador conserva la información de ausencia sin inventar un precio.

| Precio | Registros | Promedio de UNITS |
|---|---:|---:|
| Disponible (`0`) | 18.551 | 1,4016 |
| Ausente (`1`) | 1.449 | 0,0000 |

Es binaria y se puede representar directamente como 0/1. Su disponibilidad previa no está demostrada por este CSV histórico. Es una candidata condicional: solo la usaría si el sistema puede informar el precio planificado o su ausencia antes del día objetivo. Si los faltantes se generan después de observar ventas, la excluiría. Mientras no se confirme, una alternativa disponible desde el calendario es `es_fin_de_semana`, cuya evidencia aparece en el ejercicio 1. No se usan las etiquetas retrospectivas de pre-lanzamiento para fabricar este indicador.

**Característica 2: `tienda_categoria`.**

Surge de las diferencias entre tiendas observadas en Semana 8 y del conocimiento de que no todas las categorías venden igual en cada local. Une tienda y categoría para permitir un efecto específico de cada combinación.

| Combinación | Registros | Promedio de UNITS |
|---|---:|---:|
| CA_3, FOODS | 816 | 2,7279 |
| CA_3, HOBBIES | 355 | 0,6817 |
| CA_4, HOUSEHOLD | 606 | 0,4637 |

Hay 30 combinaciones en la muestra. Los promedios sugieren diferencias, pero también pueden influir la mezcla de productos y las fechas. Tienda y categoría se conocen antes de vender. Para regresión lineal usaría one-hot de esta combinación: asignarle un entero impondría un orden inexistente. No introduciría los promedios de esta tabla como predictores calculados con todos los datos; aquí solo son evidencia descriptiva.

Los resultados completos están en [resultados_features.json](./plantilla-features/resultados_features.json). Las diferencias de promedios justifican investigar estas columnas; todavía no prueban que mejoren el modelo. Eso se evaluará entrenando y validando por fecha.

## Antes de la próxima sesión: evitar fuga al escalar

Lectura: "The Dreaded Antagonist: Data Leakage in Machine Learning", Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Calcularía la media y la desviación estándar solo con entrenamiento. Luego usaría esos mismos valores para transformar validación y prueba. Si calculo las estadísticas con todos los datos, la preparación del modelo ya conocería parte del período futuro que quiero evaluar. Esta regla se explica en la lectura.

En el proyecto separaría entrenamiento y evaluación con una fecha de corte, manteniendo todas las filas de una misma fecha en el mismo conjunto. Antes de un modelo definitivo también revisaría la cobertura temporal de una muestra más representativa. Cualquier imputación, escalado o codificación que aprenda de datos debe ajustarse únicamente con entrenamiento. Para regresión, los precios faltantes requieren además una estrategia de imputación aprendida en entrenamiento, el indicador de ausencia no sustituye ese paso.
