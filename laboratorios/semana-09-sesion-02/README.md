# Semana 9, Sesión 2: pipeline reproducible de características

> Objetivo: preparar los datos de ventas con una secuencia que rellene faltantes, convierta categorías y escale números, aprendiendo solamente de entrenamiento.

Los ejercicios están adaptados al proyecto de ventas y al objetivo `UNITS`, sin usar taxis. Se reutiliza el código de la sesión 1 para leer solo las primeras 100.000 filas de `data/processed/obt_sales.csv` y elegir 20.000 con semilla 42. 

## Requisitos y ejecución

Verificado con scikit-learn 1.9.1, pandas y numpy en el entorno `.venv` del proyecto. No fue necesario instalar paquetes. Desde la raíz del repositorio, en PowerShell:

```powershell
.venv/Scripts/python.exe laboratorios/semana-09-sesion-02/plantilla-pipeline/pipeline_proyecto.py
.venv/Scripts/python.exe laboratorios/semana-09-sesion-02/plantilla-ejercicio2/pipeline_completo.py
```

Los scripts resuelven las rutas a partir de su ubicación y también funcionan al ejecutarlos desde su carpeta. Con otro entorno de Python activado, se puede sustituir `.venv/Scripts/python.exe` por `python`.

## Antes de empezar

**¿Qué es la fuga de información y por qué puede dar una evaluación excelente que falla en producción?**

Ocurre cuando el proceso aprende información que no tendría al hacer una predicción real. Puede conocer resultados futuros o estadísticas calculadas con el conjunto reservado para evaluar. La evaluación deja de ser una prueba independiente. En el proyecto, `REVENUE` del mismo día sería una pista de las unidades que queremos predecir, pero todavía no existe antes de vender.

**¿Qué forma de fuga representa usar la duración real de un viaje antes de que ocurra?**

Es fuga por información posterior al momento de predicción. La duración final no se conoce al pedir el taxi. El equivalente en ventas es usar ingresos o indicadores de falta de stock calculados con las ventas del propio día. Una correlación alta no convierte estas columnas en entradas válidas.

## Pregunta de apertura

**¿Con qué datos calculo la media y la desviación estándar para escalar?**

Solo con entrenamiento. A prueba le aplico esas mismas cifras sin volver a aprenderlas. La regla es `fit_transform(X_ent)` y después `transform(X_prueba)`. Calcularlas con todo el dataset permite que la preparación conozca el período futuro.

## Ejercicio 1: encontrar y corregir la fuga

**Respuesta 1: ¿en qué línea entra prueba al entrenamiento?**
Enunciado: starter/viajes_features.csv tiene los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. Un compañero imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. El código está en la diapositiva.

En la llamada a `fit` o `fit_transform` del imputador o del escalador sobre todos los datos, antes de dividir. La diapositiva no está en el repositorio, por lo que no se puede indicar un número de línea original.

**Respuesta 2: ¿cuánto debe cambiar el MAE al corregirlo?**

No hay una cantidad garantizada, puede aumentar, disminuir o casi no cambiar. La existencia de fuga se determina por qué datos se usaron para aprender, no por el tamaño del cambio en el error.

**Resolución y resultados reales.**

[pipeline_completo.py](./plantilla-ejercicio2/pipeline_completo.py) compara dos regresiones lineales en la misma división temporal. La versión correcta incluye preparación y regresión en un `Pipeline`, ajustado únicamente con entrenamiento. La versión incorrecta está marcada como demostración y ajusta la preparación con toda la muestra.

```python
correcto = Pipeline([
    ('preparar', preparacion_modelo()),
    ('modelo', LinearRegression()),
])
correcto.fit(X_ent, y_ent)
predicciones = correcto.predict(X_prueba)
```

| Variante | MAE de prueba, en unidades |
|---|---:|
| Preparación correcta | 1,652345 |
| Preparación con fuga deliberada | 1,652345 |

La diferencia es aproximadamente -0,000000000000025: es despreciable y compatible con redondeo numérico. En estas entradas no hay faltantes y una regresión lineal con intercepto puede compensar cambios de escala. Por eso este ejemplo no muestra una mejora visible al corregir la fuga. La versión incorrecta sigue estando mal. El MAE es un resultado didáctico sobre una muestra limitada, no una evaluación definitiva del proyecto ni una comparación con una línea base.

Para esta comparación se usa `drop='first'` en el codificador, evitando la redundancia de las categorías con el intercepto. La preparación de los ejercicios 2 y 3 conserva las 30 categorías completas para mostrar claramente el conteo de columnas.

## Durante el ConcepTest

**Voto:** 

**Razonamiento:**

**Cambio tras discutir:**
## Ejercicio 2: el pipeline completo

Solo o en pareja. Parte de la plantilla plantilla-ejercicio2/pipeline_completo.py y córrela desde su carpeta. Completa los 5 TODO: división, ColumnTransformer con tres grupos de columnas, fit solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio.


Resuelto en [pipeline_completo.py](./plantilla-ejercicio2/pipeline_completo.py), con estos cinco pasos adaptados al proyecto:

1. Separar por fecha, reservando aproximadamente el último 20% de las fechas observadas. No se divide al azar porque queremos predecir días futuros.
2. Preparar tres grupos con `ColumnTransformer`: numéricas, categóricas y columnas que pasan sin cambios.
3. Ajustar solo en entrenamiento y transformar ambos conjuntos con el mismo objeto.
4. Comparar las medianas aprendidas con las de entrenamiento y con las de la muestra completa.
5. Transformar una combinación nueva, `TIENDA_NUEVA__FOODS`, en lugar de una zona nueva de taxi.

**Respuesta 1: ¿cuántas columnas quedan?**

Quedan **35 columnas**:

| Grupo | Entradas | Salidas |
|---|---|---:|
| Numéricas: mediana y escalado | `YEAR`, `D_NUM` | 2 |
| Categóricas: moda y one-hot | `tienda_categoria` | 30 |
| Ya listas: pasan sin cambiar | `es_fin_de_semana`, `dia_sin`, `dia_cos` | 3 |
| Total | | 35 |

`YEAR` representa el año y `D_NUM` la posición del día en la serie original. Se incluyen como señales temporales simples para practicar el pipeline; su utilidad y posible redundancia deberán evaluarse después. No se usan identificadores numéricos de tienda como si representaran cantidades.

**Respuesta 2: ¿con qué medianas coinciden las aprendidas?**

| Columna | Mediana aprendida | Mediana de entrenamiento | Mediana de toda la muestra |
|---|---:|---:|---:|
| YEAR | 2016 | 2016 | 2016 |
| D_NUM | 1830 | 1830 | 1830 |

Coinciden con entrenamiento, pero en este caso también coinciden con la muestra completa. Por tanto, comparar solo las medianas no permite distinguir ambos procedimientos. La comprobación principal es que `fit_transform` recibe exclusivamente `X_ent`.

Como evidencia adicional, las medias del escalador sí distinguen los conjuntos:

| Columna | Media aprendida = media de entrenamiento | Media de toda la muestra |
|---|---:|---:|
| YEAR | 2014,818387 | 2015,005200 |
| D_NUM | 1492,307044 | 1555,711550 |


**Respuesta 3: ¿qué sucede con una tienda nueva?**

`OneHotEncoder(handle_unknown='ignore')` transforma el registro sin fallar y conserva las 35 columnas. Las 30 posiciones de `tienda_categoria` quedan en cero: no asigna la combinación nueva a ninguna de las conocidas. Esto permite procesarla, pero no significa que el modelo haya aprendido su comportamiento específico.

Con `pd.get_dummies` aplicado por separado, entrenamiento tiene 30 columnas de categorías y el registro nuevo crea solo `tienda_categoria_TIENDA_NUEVA__FOODS`. Los nombres y las dimensiones no coinciden. Se podría alinear manualmente, pero el codificador conserva el esquema aprendido de forma automática.

La evidencia está en [resultados_ejercicios.json](./plantilla-ejercicio2/resultados_ejercicios.json).

## Ejercicio 3: pipeline del proyecto

### Paso 1: clasificación y disponibilidad

- Objetivo: `UNITS`, unidades vendidas por producto, tienda y día.
- Numéricas: `YEAR` y `D_NUM`.
- Categórica: `tienda_categoria`, creada en la sesión 1 a partir de tienda y categoría.
- Ya listas: `es_fin_de_semana`, `dia_sin` y `dia_cos`, calculadas desde la fecha.
- Fecha usada para separar: `DATE`; no entra directamente como texto en el modelo.

Se reconstruyen las características con las funciones de la sesión anterior; no se usa el JSON de resultados como dataset. Crear estas columnas fila a fila antes de dividir es válido porque no aprende estadísticas de otros registros.

**Decisión sobre las dos características de la sesión anterior:** se conserva `tienda_categoria`. Se excluye `precio_no_disponible` porque no se ha demostrado que la ausencia del precio se conozca antes del día pronosticado. Se usa `es_fin_de_semana`, la alternativa disponible desde el calendario que ya se había propuesto en el README de la sesión 1. También se excluye `SELL_PRICE` hasta verificar que es un precio planificado. Esto evita convertir una suposición pendiente en una entrada del modelo.

Se dejan fuera `REVENUE`, diagnósticos de ventas anómalas, indicadores retrospectivos de stockout y `_LOADED_AT`. Tampoco se usan `IS_PRE_LAUNCH` o `IS_STORE_CLOSED` sin acreditar su disponibilidad previa. El modelo es una práctica de preparación, no una solución completa de pronóstico de demanda.

### Paso 2: división temporal

La muestra tiene solo 20 fechas distintas, pese a abarcar varios años. Se ordenan las fechas y se reservan las últimas cuatro para prueba. Todas las filas de una fecha quedan en el mismo conjunto. La fecha de corte es 2016-04-03.

| Conjunto | Regla | Filas | Columnas preparadas |
|---|---|---:|---:|
| Entrenamiento | DATE anterior a 2016-04-03 | 16.838 | 35 |
| Prueba | DATE desde 2016-04-03 | 3.162 | 35 |

Entrenamiento termina el 2016-04-02 y prueba comienza el 2016-04-03. El reparto es 84,19% y 15,81% de las filas: reservar 20% de fechas no garantiza reservar 20% de filas. No se mezclan filas futuras en entrenamiento para alcanzar una proporción exacta.

### Paso 3: construcción y ejecución

Resuelto en [pipeline_proyecto.py](./plantilla-pipeline/pipeline_proyecto.py). La preparación tiene esta estructura:

```python
preparacion = ColumnTransformer([
    ('numericas', Pipeline([
        ('imputar', SimpleImputer(strategy='median')),
        ('escalar', StandardScaler()),
    ]), NUMERICAS),
    ('categoricas', Pipeline([
        ('imputar', SimpleImputer(strategy='most_frequent')),
        ('codificar', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), CATEGORICAS),
    ('ya_listas', 'passthrough', YA_LISTAS),
])
X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
```

Salida obtenida:

```text
Entrenamiento: (16838, 35)
Prueba: (3162, 35)
```

Las columnas producidas son:

```text
numericas__YEAR
numericas__D_NUM
categoricas__tienda_categoria_CA_1__FOODS
categoricas__tienda_categoria_CA_1__HOBBIES
categoricas__tienda_categoria_CA_1__HOUSEHOLD
categoricas__tienda_categoria_CA_2__FOODS
categoricas__tienda_categoria_CA_2__HOBBIES
categoricas__tienda_categoria_CA_2__HOUSEHOLD
categoricas__tienda_categoria_CA_3__FOODS
categoricas__tienda_categoria_CA_3__HOBBIES
categoricas__tienda_categoria_CA_3__HOUSEHOLD
categoricas__tienda_categoria_CA_4__FOODS
categoricas__tienda_categoria_CA_4__HOBBIES
categoricas__tienda_categoria_CA_4__HOUSEHOLD
categoricas__tienda_categoria_TX_1__FOODS
categoricas__tienda_categoria_TX_1__HOBBIES
categoricas__tienda_categoria_TX_1__HOUSEHOLD
categoricas__tienda_categoria_TX_2__FOODS
categoricas__tienda_categoria_TX_2__HOBBIES
categoricas__tienda_categoria_TX_2__HOUSEHOLD
categoricas__tienda_categoria_TX_3__FOODS
categoricas__tienda_categoria_TX_3__HOBBIES
categoricas__tienda_categoria_TX_3__HOUSEHOLD
categoricas__tienda_categoria_WI_1__FOODS
categoricas__tienda_categoria_WI_1__HOBBIES
categoricas__tienda_categoria_WI_1__HOUSEHOLD
categoricas__tienda_categoria_WI_2__FOODS
categoricas__tienda_categoria_WI_2__HOBBIES
categoricas__tienda_categoria_WI_2__HOUSEHOLD
categoricas__tienda_categoria_WI_3__FOODS
categoricas__tienda_categoria_WI_3__HOBBIES
categoricas__tienda_categoria_WI_3__HOUSEHOLD
ya_listas__es_fin_de_semana
ya_listas__dia_sin
ya_listas__dia_cos
```

En [resultados_pipeline.json](./plantilla-pipeline/resultados_pipeline.json) quedan las dimensiones, nombres, estadísticas aprendidas, fechas y versión de scikit-learn. Se comprueban separación temporal, igualdad del esquema, estadísticas de entrenamiento y ausencia de valores no finitos.

## Limitaciones de la práctica

La muestra se selecciona aleatoriamente solo dentro del primer bloque físico, no de todo el archivo. Contiene pocas fechas y no representa una serie diaria completa. Sirve para comprobar que el código funciona por lo que no permite afirmar que el error medido represente el comportamiento futuro de todo el negocio. No se construyen rezagos ni promedios móviles con días ausentes. Para evaluar el modelo definitivo haría falta una muestra temporal más adecuada, un período de validación y una comparación con un pronóstico sencillo de referencia.

## Antes de la próxima sesión

Lectura indicada por el curso: [Ensembles in Machine Learning](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/).

**Si un modelo tiene un error mucho menor en entrenamiento que en prueba, ¿qué le pasa?**

Es una señal de sobreajuste: aprendió demasiado los detalles del entrenamiento y funciona peor con datos nuevos. En una serie temporal también revisaría si cambió el comportamiento entre períodos. Compararía con un modelo más sencillo y ajustaría la complejidad usando validación temporal, sin elegir repetidamente cambios mirando el conjunto final de prueba.
