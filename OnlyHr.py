import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/sould/OneDrive/Documents/GitHub/Medicheckk/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    return faces

def butter_bandpass_filter(data, order=4, lowcut=0.7, highcut=2.0, fs=30):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_hr(frames, face, fps):
    (x, y, w, h) = face
    intensity_values = []
    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        b, g, r = cv2.split(roi)
        avg_intensity = np.mean(g)  # Usa la intensidad promedio del canal verde
        intensity_values.append(avg_intensity)  # Añade al array las intensidades promedio
    
    # Aplica el filtro pasabandas si hay suficiente información
    if len(intensity_values) > 30:
        filtered_intensity_values = butter_bandpass_filter(intensity_values, order=4, lowcut=0.7, highcut=2.0, fs=fps)
    else:
        filtered_intensity_values = intensity_values
    
    # Encuentra los picos de los valores filtrados
    peaks, _ = find_peaks(filtered_intensity_values, distance=fps*0.450)
    
    # Extrae los tiempos de los picos
    peak_times = np.array(peaks) / fps
    if len(peak_times) < 2:
        return 0, intensity_values, peaks
    
    intervals = np.diff(peak_times)
    
    # Calcula la frecuencia cardíaca
    avg_interval = np.mean(intervals)
    if avg_interval == 0:
        return 0, intensity_values, peaks
    
    hr = 60 / avg_interval
    
    return hr, filtered_intensity_values, peaks

import matplotlib.pyplot as plt

def plot_hr(intensity_values, filtered_intensity_values, peaks):
    # Plot the original intensity in one window
    plt.figure()  # Create a new figure for the original intensity
    plt.plot(intensity_values, label="Intensidad Original")
    plt.xlabel("Frame")
    plt.ylabel("Intensidad")
    plt.title("Intensidad Original")
    plt.legend()
    
    # Plot the filtered intensity and peaks in another window   
    plt.figure()  # Create a new figure for the filtered intensity and peaks
    plt.plot(filtered_intensity_values, label="Intensidad Filtrada", linestyle='--')
    plt.plot(peaks, [filtered_intensity_values[i] for i in peaks], "x", label="Picos")
    plt.xlabel("Frame")
    plt.ylabel("Intensidad")
    plt.title("Intensidad Filtrada con Picos")
    plt.legend()
    
    # Show both figures
    plt.show()

def main():
    cap = cv2.VideoCapture(0)
    fps = 30  # 30 frames por segundo
    frames = []
    frame_count = 0
    max_frames = 210
    plt.ion()  # Activa modo interactivo para gráficos en tiempo real
    
    # Captura todos los frames
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_face(frame)
        frames.append(frame)

        # Dibuja el cuadro azul alrededor de la cara detectada
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Rectángulo azul para la cara detectada
        
        # Muestra el contador de frames en pantalla
        cv2.putText(frame, f"Frames: {frame_count}/{max_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
        frame_count += 1
        key = cv2.waitKey(1)
        if key == ord('Q') or key == ord('q'):  # Finaliza si se presiona 'Q' o 'q'
            break
    
    # Procesa los frames capturados
    if frames:
        for face in detect_face(frames[0]):
            hr, filtered_intensity_values, peaks = calculate_hr(frames, face, fps)
            
            # Guarda los resultados en un archivo CSV
            df = pd.DataFrame({"Heart rate": [hr]})
            df.to_csv("data.csv", index=False)
            
            # Genera el gráfico
            plot_hr([np.mean(cv2.split(f)[1]) for f in frames], filtered_intensity_values, peaks)
            
            # Detenemos el modo interactivo para que el gráfico se mantenga visible
            plt.ioff()
            plt.show()  # Muestra el gráfico final
    
    print("Su ritmo cardíaco es:", hr)
    # Pausa para permitir que el usuario vea el gráfico
    if key == ord('Q') or key == ord('q'):  # Finaliza si se presiona 'Q' o 'q'
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

