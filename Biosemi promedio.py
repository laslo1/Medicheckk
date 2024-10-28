import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Función para aplicar un filtro de banda pasante
def filtrar_banda(data, sfreq, low_freq, high_freq):
    low_freq=0.5
    high_freq =2.0
    return mne.filter.filter_data(data, sfreq, l_freq=low_freq, h_freq=high_freq)

# Cargar archivo BDF
archivo_bdf = "C:/Users/sould/Downloads/New folder (4)/Biosemi/Biosemi/Testdata_2.bdf"

try:
    # Leer el archivo BDF con preload
    raw = mne.io.read_raw_bdf(archivo_bdf, preload=True)

    # Mostrar información básica del archivo
    print(raw.info)

    # Extraer datos (solo el primer canal, por ejemplo)
    datos, tiempos = raw[:, :]  # Extraer todos los canales y muestras
    canal_0 = datos[0, :]  # Trabajar con el primer canal de ejemplo

    # Frecuencia de muestreo
    sfreq = raw.info['sfreq'] #2048

    # Aplicar filtro de banda pasante (frecuencias del rango de latidos cardíacos: 0.5 a 4 Hz)
    canal_0_filtrado = filtrar_banda(canal_0, sfreq, low_freq=0.5, high_freq=2.0)

    # Detectar los picos en toda la señal filtrada
    picos, _ = find_peaks(canal_0_filtrado, distance=sfreq/2)  # Al menos 0.5 seg entre picos

    # Graficar la señal filtrada y los picos detectados
    plt.figure(figsize=(12, 6))
    plt.plot(tiempos, canal_0_filtrado, label="Filtrado (0.5-2 Hz)", color='blue')
    plt.plot(tiempos[picos], canal_0_filtrado[picos], "x", label="Latidos detectados", color='red')
    plt.title("Señal Filtrada y Latidos Detectados")
    plt.xlabel("Tiempo (segundos)")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calcular frecuencia cardíaca en intervalos deslizantes de 2 segundos
    intervalos = 5.9999999 # Duración de los intervalos en segundos
    desplazamiento = 1. # Desplazamiento en segundos
    desplazamientofloat= float(desplazamiento)
    hr_list = []  # Lista para almacenar las frecuencias cardíacas

    # Asegurarse de que hay suficiente tiempo para los intervalos deseados
    total_time = tiempos[-1]

    # Calcular las frecuencias cardíacas para cada intervalo deslizante
    for inicio in np.arange(0, int(total_time) - intervalos + 1, desplazamientofloat):  
        fin = inicio + intervalos  # Calcular el tiempo final del intervalo
        
        # Seleccionar la parte de la señal correspondiente al intervalo
        indices_intervalo = np.where((tiempos >= inicio) & (tiempos < fin))[0]
        datos_intervalo = canal_0_filtrado[indices_intervalo]

        # Detectar los picos en el intervalo actual
        picos_intervalo, _ = find_peaks(datos_intervalo, distance=sfreq/2)  # Al menos 0.5 seg entre picos

        # Calcular la frecuencia cardíaca si se detectan picos
        if len(picos_intervalo) > 0:
            peak_times = np.array(picos_intervalo) / sfreq  # Convertir índices de picos a tiempo
            intervals = np.diff(peak_times)
            avg_interval = np.mean(intervals)
            hr = 60 / avg_interval
        else:
            hr = 0  # No se detectaron picos, frecuencia cardíaca es 0
        
        hr_list.append(hr)  # Almacenar la frecuencia cardíaca

    # Mostrar las frecuencias cardíacas calculadas
    for i, hr in enumerate(hr_list):
        print(f"Intervalo {i + 1}: Frecuencia cardíaca = {hr:.2f} bpm")

    # Calcular el promedio de frecuencia cardíaca
    if hr_list:
        promedio_hr = np.mean(hr_list)
        print(f"Promedio de Frecuencia Cardíaca = {promedio_hr:.2f} bpm")

        # Graficar la frecuencia cardíaca
        plt.figure(figsize=(12, 6))
        plt.plot(hr_list, label="Frecuencia Cardíaca (bpm)", color='blue')

        # Graficar el promedio de frecuencia cardíaca
        for i, hr in enumerate(hr_list):
            plt.scatter(i, hr, color='orange')  # Agregar puntos del promedio
        plt.axhline(promedio_hr, color='orange', linestyle='--', label='Promedio de Frecuencia Cardíaca')
        plt.title("Frecuencia Cardíaca")
        plt.xlabel("Intervalo de Tiempo (deslizante)")
        plt.ylabel("Frecuencia Cardíaca (bpm)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No se detectaron frecuencias cardíacas.")

except Exception as e:
    print(f"Error al leer el archivo: {e}")
