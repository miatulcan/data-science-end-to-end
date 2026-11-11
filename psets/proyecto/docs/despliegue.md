# Despliegue del pronóstico de ventas

## Patrón elegido

Elijo predicción por lotes. La predicción ayuda al equipo de inventario a
planificar las cantidades que solicitará para cada tienda y categoría en los
días siguientes. El equipo puede esperar hasta una noche para recibir el
pronóstico de la siguiente jornada, así que la latencia objetivo es de menos
de 24 horas.

## Casos que se predicen

Se predice una combinación de fecha, tienda y categoría. Las fechas futuras y
las combinaciones habilitadas de tienda y categoría se conocen antes de
ejecutar el lote. Por eso se pueden generar todas las predicciones juntas y
guardarlas para que el equipo las consulte después.

## Datos disponibles

Las variables usadas por el pipeline son el año, el número de día del año,
la combinación de tienda y categoría, y las características de calendario.
La fecha futura, la tienda y la categoría están disponibles antes de hacer el
pronóstico. No uso `SELL_PRICE` ni `precio_no_disponible` porque no está
confirmado que se conozcan antes de la fecha pronosticada.

## Qué se pierde al descartar el patrón online

El equipo no recibe una predicción inmediata para una combinación nueva que
aparezca después de generar el lote. Tendría que esperar al siguiente lote o
solicitar una ejecución extraordinaria. Para la planificación diaria, esa
espera es aceptable y evita mantener un servicio de predicción en línea.
