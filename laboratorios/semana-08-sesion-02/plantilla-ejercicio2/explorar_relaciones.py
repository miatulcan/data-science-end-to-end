import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# 1. QUITAR SOLO LOS ERRORES CONFIRMADOS
# Conservamos las tarifas altas que pueden corresponder a viajes válidos.
limpios = viajes.query("tarifa != 999.99 and duracion_min >= 0")
print("Viajes originales:", len(viajes))
print("Viajes después de quitar los errores:", len(limpios))

# 2. EXPLORAR DISTANCIA CONTRA TARIFA
plt.figure()
plt.scatter(limpios["distancia_km"], limpios["tarifa"], alpha=0.4, s=12)
plt.title("Distancia contra tarifa, sin errores confirmados")
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.tight_layout()
plt.savefig("figuras/distancia_vs_tarifa.png")
plt.close()

# La banda horizontal de 70 puede indicar una tarifa fija: revisamos sus rutas.
tarifa_70 = limpios.query("tarifa == 70")
print("\nViajes con tarifa de 70:", len(tarifa_70))
print("Distancia mínima y máxima de esos viajes:")
print(tarifa_70["distancia_km"].agg(["min", "max"]))
print("Rutas de los viajes con tarifa de 70:")
print(tarifa_70.groupby(["zona_origen", "zona_destino"]).size())

# Una distancia cero con cobro también merece revisión, no eliminación automática.
print("\nViajes con distancia cero:", len(limpios.query("distancia_km == 0")))

# 3. COMPARAR LA PROPINA POR MÉTODO DE PAGO
# dropna=False incluye los viajes cuyo método de pago está vacío.
resumen_propinas = limpios.groupby("metodo_pago", dropna=False).agg(
    viajes=("propina", "size"),
    propina_promedio=("propina", "mean"),
    propinas_cero=("propina", lambda valores: (valores == 0).sum()),
)
print("\nPropinas por método de pago:")
print(resumen_propinas)

# 4. MEDIR LA RELACIÓN LINEAL CON LA TARIFA
# Seleccionamos las variables relevantes: id_viaje es solo un identificador.
columnas = ["distancia_km", "duracion_min", "pasajeros", "propina", "tarifa"]
correlaciones = limpios[columnas].corr()
print("\nCorrelaciones de Pearson con la tarifa:")
print(correlaciones["tarifa"].drop("tarifa"))

# 5. REVISAR PASAJEROS VACÍOS Y EN CERO
# Un dato faltante y un cero son casos distintos y se cuentan por separado.
print("\nPasajeros vacíos:", limpios["pasajeros"].isna().sum())
print("Pasajeros en cero:", len(limpios.query("pasajeros == 0")))
