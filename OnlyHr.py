import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import mne  # Asegúrate de tener la librería MNE instalada para manejar archivos BDF

def butter_bandpass_filter(data, order=4, lowcut=0.7, highcut=2.0, fs=2048):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_hr(signal, fs):
    # Aplica el filtro pasabandas a la señal
    filtered_signal = butter_bandpass_filter(signal, order=4, lowcut=0.7, highcut=2.0, fs=fs)
    
    # Encuentra los picos de la señal filtrada
    peaks, _ = find_peaks(filtered_signal, distance=fs*0.450)
    
    # Extrae los tiempos de los picos
    peak_times = np.array(peaks) / fs
    if len(peak_times) < 2:
        return 0, filtered_signal, peaks
    
    intervals = np.diff(peak_times)
    
    # Calcula la frecuencia cardíaca
    avg_interval = np.mean(intervals)
    if avg_interval == 0:
        return 0, filtered_signal, peaks
    
    hr = 60 / avg_interval
    
    return hr, filtered_signal, peaks

def plot_hr(signal, filtered_signal, peaks):
    # Grafica la señal original y la señal filtrada
    plt.figure()
    plt.plot(signal, label="Señal Original")
    plt.plot(filtered_signal, label="Señal Filtrada", linestyle='--')
    plt.plot(peaks, filtered_signal[peaks], "x", label="Picos")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.title("Señal Filtrada con Picos")
    plt.legend()
    plt.show()

def main():
    # Carga el archivo BDF
    raw = mne.io.read_raw_bdf('ruta/a/tu/archivo.bdf', preload=True)
    
    # Extrae la señal (suponiendo que estás usando el canal correcto)
    # Asegúrate de elegir el canal que contenga la señal de ritmo cardíaco
    signal = raw.get_data(picks=['nombre_del_canal'])[0]  # Reemplaza 'nombre_del_canal' con el nombre correcto
    fs = raw.info['sfreq']  # Frecuencia de muestreo del archivo BDF

    hr, filtered_signal, peaks = calculate_hr(signal, fs)

    # Guarda los resultados en un archivo CSV
    df = pd.DataFrame({"Heart rate": [hr]})
    df.to_csv("data.csv", index=False)

    # Genera el gráfico
    plot_hr(signal, filtered_signal, peaks)

    print("Su ritmo cardíaco es:", hr)

if __name__ == "__main__":
    main()
