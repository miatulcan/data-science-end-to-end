from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Objetivo asumido: predecir unidades vendidas por producto, tienda y día.
# Fuente: exportación obt_sales.csv compartida por el compañero de la Semana 7.
# Los indicadores de stockout se exploran como señales, no como causas verificadas.
# El CSV supera los 13 GB: lo recorremos por bloques, sin cargarlo completo.
carpeta = Path(__file__).resolve().parent
archivo = carpeta.parents[2] / 'data' / 'processed' / 'obt_sales.csv'
figuras = carpeta / 'figuras'
figuras.mkdir(exist_ok=True)
OBJETIVO = 'UNITS'
filas = 0
faltantes = None
frecuencias = pd.Series(dtype='int64')
resumen_tiendas = pd.DataFrame()
conteos = dict(ceros=0, negativos=0, pre_lanzamiento=0, ceros_pre_lanzamiento=0,
               cierre=0, ceros_cierre=0, stockout=0, ceros_stockout=0,
               precio_faltante_activo=0, ingresos_comparables=0, ingresos_inconsistentes=0)
muestras = []
fecha_min = None
fecha_max = None

# 1. RESUMEN COMPLETO: los contadores se acumulan para todo el archivo.
for bloque in pd.read_csv(archivo, chunksize=200_000, low_memory=False):
    filas += len(bloque)
    faltantes_bloque = bloque.isna().sum()
    faltantes = faltantes_bloque if faltantes is None else faltantes.add(faltantes_bloque)
    frecuencias = frecuencias.add(bloque[OBJETIVO].value_counts(), fill_value=0)
    ceros = bloque[OBJETIVO].eq(0)
    pre = bloque['IS_PRE_LAUNCH'].eq(True)
    cierre = bloque['IS_STORE_CLOSED'].eq(True)
    stock = bloque['IS_SUSPECTED_STOCKOUT_CONSERVATIVE'].eq(True)
    conteos['ceros'] += int(ceros.sum())
    conteos['negativos'] += int(bloque[OBJETIVO].lt(0).sum())
    for nombre, mascara in [('pre_lanzamiento', pre), ('cierre', cierre), ('stockout', stock)]:
        conteos[nombre] += int(mascara.sum())
        conteos['ceros_' + nombre] += int((ceros & mascara).sum())
    conteos['precio_faltante_activo'] += int((bloque['SELL_PRICE'].isna() & ~pre & ~cierre).sum())
    comparables = bloque[['UNITS', 'SELL_PRICE', 'REVENUE']].notna().all(axis=1)
    diferencia = (bloque['REVENUE'] - bloque['UNITS'] * bloque['SELL_PRICE']).abs()
    conteos['ingresos_comparables'] += int(comparables.sum())
    conteos['ingresos_inconsistentes'] += int((comparables & diferencia.gt(0.011)).sum())
    tiendas = bloque.groupby('STORE_ID')[OBJETIVO].agg(['sum', 'count'])
    resumen_tiendas = resumen_tiendas.add(tiendas, fill_value=0)
    minimo, maximo = bloque['DATE'].min(), bloque['DATE'].max()
    fecha_min = minimo if fecha_min is None else min(fecha_min, minimo)
    fecha_max = maximo if fecha_max is None else max(fecha_max, maximo)
    # Igual fracción por bloque; muestra reproducible para visualizar precio y ventas.
    muestras.append(bloque[['SELL_PRICE', 'UNITS']].sample(frac=0.001, random_state=42))
    if filas % 2_000_000 == 0:
        print('Filas procesadas:', filas, flush=True)

# 2. DISTRIBUCIÓN E IQR EXACTOS, a partir de las frecuencias acumuladas.
frecuencias = frecuencias.sort_index().astype('int64')
acumuladas = frecuencias.cumsum()
n = int(frecuencias.sum())
def cuantil(p):
    posicion = (n - 1) * p
    inferior = int(posicion)
    superior = inferior if posicion == inferior else inferior + 1
    a = float(acumuladas.index[acumuladas.searchsorted(inferior + 1)])
    b = float(acumuladas.index[acumuladas.searchsorted(superior + 1)])
    return a + (posicion - inferior) * (b - a)

q1, mediana, q3 = cuantil(0.25), cuantil(0.5), cuantil(0.75)
iqr = q3 - q1
limite_inferior, limite_superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
atipicos = int(frecuencias[(frecuencias.index < limite_inferior) | (frecuencias.index > limite_superior)].sum())
media = float((frecuencias.index * frecuencias).sum() / n)
plt.figure(figsize=(8, 4))
plt.hist(frecuencias.index, weights=frecuencias.values, bins=40)
plt.yscale('log')
plt.xlabel('Unidades vendidas por producto, tienda y día')
plt.ylabel('Registros (escala logarítmica)')
plt.title('Distribución de UNITS: archivo completo')
plt.tight_layout()
plt.savefig(figuras / 'distribucion_objetivo.png')
plt.close()

# 3. RELACIONES: precio en una muestra y promedio por tienda en todo el CSV.
muestra = pd.concat(muestras, ignore_index=True)
plt.figure(figsize=(8, 4))
plt.scatter(muestra['SELL_PRICE'], muestra['UNITS'], alpha=0.15, s=5)
plt.xlabel('Precio de venta')
plt.ylabel('Unidades vendidas')
plt.title('Precio contra ventas: muestra del 0.1 %')
plt.tight_layout()
plt.savefig(figuras / 'precio_vs_unidades.png')
plt.close()
resumen_tiendas['promedio_unidades'] = resumen_tiendas['sum'] / resumen_tiendas['count']
resumen_tiendas['promedio_unidades'].plot.bar(figsize=(8, 4))
plt.ylabel('Unidades promedio por registro')
plt.title('Ventas por tienda: archivo completo')
plt.tight_layout()
plt.savefig(figuras / 'ventas_por_tienda.png')
plt.close()

# Guardamos evidencia pequeña y reproducible; el CSV original no se modifica.
resultado = dict(objetivo=OBJETIVO, filas=filas, columnas=len(faltantes),
    fecha_min=fecha_min, fecha_max=fecha_max, faltantes={k:int(v) for k,v in faltantes.items()},
    minimo=float(frecuencias.index.min()), maximo=float(frecuencias.index.max()),
    media=media, mediana=mediana, q1=q1, q3=q3, iqr=iqr,
    limite_inferior=limite_inferior, limite_superior=limite_superior, atipicos=atipicos,
    conteos=conteos, muestra_filas=len(muestra),
    correlacion_precio_muestra=float(muestra[['SELL_PRICE','UNITS']].corr().loc['SELL_PRICE','UNITS']),
    tiendas=resumen_tiendas.to_dict(orient='index'))
(carpeta / 'resultados_eda.json').write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(resultado, indent=2, ensure_ascii=False), flush=True)
